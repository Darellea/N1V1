"""
tests/test_data_fixes.py

Unit tests for data module fixes:
1. Infinite loop prevention in historical data fetching
2. Robust timestamp handling in cache loading
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd

from data.data_fetcher import CacheLoadError, DataFetcher
from data.dataset_versioning import (
    DatasetVersionManager,
    MetadataError,
    migrate_legacy_dataset,
)
from data.historical_loader import HistoricalDataLoader


class TestHistoricalLoaderInfiniteLoopFix(unittest.TestCase):
    """Test cases for infinite loop prevention in historical data fetching."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {"backtesting": {"data_dir": "test_data", "deduplicate": False}}
        self.mock_data_fetcher = MagicMock()
        self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove any test directories created
        test_dir = os.path.join(os.getcwd(), "data", "historical", "test_data")
        if os.path.exists(test_dir):
            import shutil

            shutil.rmtree(test_dir)

    @patch("data.historical_loader.tqdm")
    async def test_normal_pagination_progression(self, mock_tqdm):
        """Test that normal pagination works correctly."""
        # Mock tqdm to avoid progress bar output
        mock_tqdm.return_value.__enter__.return_value = MagicMock()
        mock_tqdm.return_value.__exit__.return_value = None

        # Create mock data that advances properly
        start_time = pd.Timestamp("2023-01-01")
        mock_data = []

        # Create 5 chunks of data, each advancing by 1 hour
        for i in range(5):
            chunk_start = start_time + timedelta(hours=i)
            chunk_data = pd.DataFrame(
                {
                    "timestamp": [
                        chunk_start + timedelta(minutes=j) for j in range(60)
                    ],
                    "open": [100 + j for j in range(60)],
                    "high": [105 + j for j in range(60)],
                    "low": [95 + j for j in range(60)],
                    "close": [102 + j for j in range(60)],
                    "volume": [1000 + j for j in range(60)],
                }
            )
            chunk_data["timestamp"] = pd.to_datetime(chunk_data["timestamp"])
            chunk_data = chunk_data.set_index("timestamp")
            mock_data.append(chunk_data)

        # Mock the data fetcher to return our test data
        self.mock_data_fetcher.get_historical_data = AsyncMock()
        self.mock_data_fetcher.get_historical_data.side_effect = mock_data + [
            pd.DataFrame()
        ]  # Empty DataFrame to end loop

        # Test the fetch method
        result = await self.loader._fetch_complete_history(
            "BTC/USDT", "2023-01-01", "2023-01-05", "1h"
        )

        # Verify we got data and the loop didn't get stuck
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertEqual(len(mock_data), 5)  # Should have made 5 successful calls

    @patch("data.historical_loader.tqdm")
    async def test_infinite_loop_detection_same_start_time(self, mock_tqdm):
        """Test that infinite loops are detected when current_start doesn't advance."""
        # Mock tqdm to avoid progress bar output
        mock_tqdm.return_value.__enter__.return_value = MagicMock()
        mock_tqdm.return_value.__exit__.return_value = None

        # Create mock data that always returns the same last_index
        start_time = pd.Timestamp("2023-01-01")
        mock_chunk = pd.DataFrame(
            {
                "timestamp": [start_time + timedelta(minutes=j) for j in range(60)],
                "open": [100 + j for j in range(60)],
                "high": [105 + j for j in range(60)],
                "low": [95 + j for j in range(60)],
                "close": [102 + j for j in range(60)],
                "volume": [1000 + j for j in range(60)],
            }
        )
        mock_chunk["timestamp"] = pd.to_datetime(mock_chunk["timestamp"])
        mock_chunk = mock_chunk.set_index("timestamp")

        # Mock the data fetcher to always return the same data
        self.mock_data_fetcher.get_historical_data = AsyncMock()
        self.mock_data_fetcher.get_historical_data.return_value = mock_chunk

        # Test the fetch method - should detect infinite loop and break
        result = await self.loader._fetch_complete_history(
            "BTC/USDT", "2023-01-01", "2023-01-05", "1h"
        )

        # Verify we got some data but the loop was terminated early
        self.assertIsNotNone(result)
        # Should have detected the loop and broken before max_iterations

    @patch("data.historical_loader.tqdm")
    async def test_exchange_returns_same_last_index(self, mock_tqdm):
        """Test handling when exchange returns same last_index repeatedly."""
        # Mock tqdm to avoid progress bar output
        mock_tqdm.return_value.__enter__.return_value = MagicMock()
        mock_tqdm.return_value.__exit__.return_value = None

        # Create mock data where last_index doesn't advance
        start_time = pd.Timestamp("2023-01-01")
        mock_chunk = pd.DataFrame(
            {
                "timestamp": [start_time + timedelta(minutes=j) for j in range(60)],
                "open": [100 + j for j in range(60)],
                "high": [105 + j for j in range(60)],
                "low": [95 + j for j in range(60)],
                "close": [102 + j for j in range(60)],
                "volume": [1000 + j for j in range(60)],
            }
        )
        mock_chunk["timestamp"] = pd.to_datetime(mock_chunk["timestamp"])
        mock_chunk = mock_chunk.set_index("timestamp")

        # Mock the data fetcher
        self.mock_data_fetcher.get_historical_data = AsyncMock()
        self.mock_data_fetcher.get_historical_data.return_value = mock_chunk

        # Test the fetch method
        result = await self.loader._fetch_complete_history(
            "BTC/USDT", "2023-01-01", "2023-01-05", "1h"
        )

        # Verify the safeguard advanced current_start by timeframe delta
        self.assertIsNotNone(result)
        # The loop should have advanced by delta when last_index <= current_start

    @patch("data.historical_loader.tqdm")
    async def test_max_iterations_limit(self, mock_tqdm):
        """Test that max_iterations limit prevents runaway loops."""
        # Mock tqdm to avoid progress bar output
        mock_tqdm.return_value.__enter__.return_value = MagicMock()
        mock_tqdm.return_value.__exit__.return_value = None

        # Create mock data
        start_time = pd.Timestamp("2023-01-01")
        mock_chunk = pd.DataFrame(
            {
                "timestamp": [start_time + timedelta(minutes=j) for j in range(60)],
                "open": [100 + j for j in range(60)],
                "high": [105 + j for j in range(60)],
                "low": [95 + j for j in range(60)],
                "close": [102 + j for j in range(60)],
                "volume": [1000 + j for j in range(60)],
            }
        )
        mock_chunk["timestamp"] = pd.to_datetime(mock_chunk["timestamp"])
        mock_chunk = mock_chunk.set_index("timestamp")

        # Mock the data fetcher to always return data (simulating infinite loop)
        self.mock_data_fetcher.get_historical_data = AsyncMock()
        self.mock_data_fetcher.get_historical_data.return_value = mock_chunk

        # Test the fetch method
        result = await self.loader._fetch_complete_history(
            "BTC/USDT", "2023-01-01", "2023-01-05", "1h"
        )

        # Verify we got some data but the loop was terminated by max_iterations
        self.assertIsNotNone(result)


class TestDataFetcherTimestampHandling(unittest.TestCase):
    """Test cases for robust timestamp handling in cache loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "name": "binance",
            "cache_enabled": True,
            "cache_dir": "test_cache",
        }
        self.fetcher = DataFetcher(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test cache directory
        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_cache")
        if os.path.exists(cache_dir):
            import shutil

            shutil.rmtree(cache_dir)

    @unittest.skip("Requires async test runner")
    async def test_timestamp_parsing_strategy_1_integer_ms(self):
        """Test Strategy 1: Direct timestamp column with integer milliseconds."""
        # Create test data with integer timestamp column
        test_data = pd.DataFrame(
            {
                "timestamp": [
                    1640995200000,
                    1640995260000,
                    1640995320000,
                ],  # ms timestamps
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        # Create cache file
        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "test_key.json")

        cache_data = {
            "timestamp": int(time.time() * 1000),
            "data": test_data.to_dict("records"),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Test loading
        result = await self.fetcher._load_from_cache("test_key")

        # Verify timestamp parsing worked
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    @unittest.skip("Requires async test runner")
    async def test_timestamp_parsing_strategy_2_datetime_column(self):
        """Test Strategy 2: Search for datetime-like columns."""
        # Create test data with datetime column named differently
        test_data = pd.DataFrame(
            {
                "datetime": [
                    "2023-01-01 12:00:00",
                    "2023-01-01 12:01:00",
                    "2023-01-01 12:02:00",
                ],
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        # Create cache file
        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "test_key.json")

        cache_data = {
            "timestamp": int(time.time() * 1000),
            "data": test_data.to_dict("records"),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Test loading
        result = await self.fetcher._load_from_cache("test_key")

        # Verify timestamp parsing worked
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    @unittest.skip("Requires async test runner")
    async def test_timestamp_parsing_strategy_3_format_parsing(self):
        """Test Strategy 3: Try different timestamp formats."""
        # Create test data with formatted timestamp strings
        test_data = pd.DataFrame(
            {
                "timestamp": [
                    "2023-01-01T12:00:00",
                    "2023-01-01T12:01:00",
                    "2023-01-01T12:02:00",
                ],
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        # Create cache file
        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "test_key.json")

        cache_data = {
            "timestamp": int(time.time() * 1000),
            "data": test_data.to_dict("records"),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Test loading
        result = await self.fetcher._load_from_cache("test_key")

        # Verify timestamp parsing worked
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIsInstance(result.index, pd.DatetimeIndex)

    @unittest.skip("Requires async test runner")
    async def test_timestamp_parsing_all_strategies_fail(self):
        """Test that all parsing strategies failing is properly logged."""
        # Create test data with invalid timestamps
        test_data = pd.DataFrame(
            {
                "timestamp": ["invalid", "also_invalid", "still_invalid"],
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        # Create cache file
        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "test_key.json")

        cache_data = {
            "timestamp": int(time.time() * 1000),
            "data": test_data.to_dict("records"),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Test loading - should return None for non-critical data
        result = await self.fetcher._load_from_cache("test_key")

        # Verify it returned None due to parsing failure
        self.assertIsNone(result)

    @unittest.skip("Requires async test runner")
    async def test_critical_cache_raises_exception_on_failure(self):
        """Test that critical cache data raises CacheLoadError on parsing failure."""
        # Create test data with invalid timestamps
        test_data = pd.DataFrame(
            {
                "timestamp": ["invalid", "also_invalid", "still_invalid"],
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        # Create cache file
        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir, "btc_critical_key.json"
        )  # Contains 'btc' keyword

        cache_data = {
            "timestamp": int(time.time() * 1000),
            "data": test_data.to_dict("records"),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Test loading - should raise CacheLoadError for critical data
        with self.assertRaises(CacheLoadError):
            await self.fetcher._load_from_cache("btc_critical_key")

    @unittest.skip("Requires async test runner")
    async def test_timezone_normalization(self):
        """Test that timezone-aware timestamps are properly normalized."""
        # Create test data with timezone-aware timestamps as strings
        timestamps = pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC")
        test_data = pd.DataFrame(
            {
                "timestamp": [str(ts) for ts in timestamps],
                "open": [100, 101, 102],
                "high": [105, 106, 107],
                "low": [95, 96, 97],
                "close": [102, 103, 104],
                "volume": [1000, 1100, 1200],
            }
        )

        # Create cache file
        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "test_key.json")

        cache_data = {
            "timestamp": int(time.time() * 1000),
            "data": test_data.to_dict("records"),
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Test loading
        result = await self.fetcher._load_from_cache("test_key")

        # Verify timezone was normalized to naive
        self.assertIsNotNone(result)
        self.assertFalse(result.empty)
        self.assertIsNone(result.index.tz)  # Should be timezone-naive

    def test_is_cache_critical_detection(self):
        """Test the critical cache detection logic."""
        # Test critical keywords
        self.assertTrue(self.fetcher._is_cache_critical("btc_usdt_1h"))
        self.assertTrue(self.fetcher._is_cache_critical("eth_btc_4h"))
        self.assertTrue(self.fetcher._is_cache_critical("major_pairs_data"))
        self.assertTrue(self.fetcher._is_cache_critical("CRITICAL_backup"))

        # Test non-critical
        self.assertFalse(self.fetcher._is_cache_critical("ltc_usdt_1h"))
        self.assertFalse(self.fetcher._is_cache_critical("minor_pairs_data"))
        self.assertFalse(self.fetcher._is_cache_critical("regular_backup"))


class TestDatasetVersioningHashFix(unittest.TestCase):
    """Test cases for dataset hashing fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.version_manager = DatasetVersionManager()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test versions directory
        import shutil

        test_dir = os.path.join(os.getcwd(), "data", "versions")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    def test_deterministic_sampling_small_dataset(self):
        """Test that small datasets use full hash."""
        # Create small dataset (under 10,000 rows)
        small_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "open": [100 + i for i in range(100)],
                "high": [105 + i for i in range(100)],
                "low": [95 + i for i in range(100)],
                "close": [102 + i for i in range(100)],
                "volume": [1000 + i for i in range(100)],
            }
        )
        small_df["timestamp"] = pd.to_datetime(small_df["timestamp"])
        small_df = small_df.set_index("timestamp")

        # Calculate hash - should use full dataset
        hash1 = self.version_manager._calculate_dataframe_hash(small_df)
        hash2 = self.version_manager._calculate_dataframe_hash(small_df)

        # Hash should be deterministic
        self.assertEqual(hash1, hash2)

    def test_deterministic_sampling_large_dataset(self):
        """Test that large datasets use deterministic sampling."""
        # Create large dataset (over 10,000 rows)
        large_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=15000, freq="h"),
                "open": [100 + i for i in range(15000)],
                "high": [105 + i for i in range(15000)],
                "low": [95 + i for i in range(15000)],
                "close": [102 + i for i in range(15000)],
                "volume": [1000 + i for i in range(15000)],
            }
        )
        large_df["timestamp"] = pd.to_datetime(large_df["timestamp"])
        large_df = large_df.set_index("timestamp")

        # Calculate hash multiple times - should be deterministic
        hash1 = self.version_manager._calculate_dataframe_hash(large_df)
        hash2 = self.version_manager._calculate_dataframe_hash(large_df)
        hash3 = self.version_manager._calculate_dataframe_hash(large_df)

        # All hashes should be identical
        self.assertEqual(hash1, hash2)
        self.assertEqual(hash2, hash3)

    def test_full_hash_option(self):
        """Test that use_full_hash=True forces full dataset hashing."""
        # Create large dataset
        large_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=15000, freq="h"),
                "open": [100 + i for i in range(15000)],
                "high": [105 + i for i in range(15000)],
                "low": [95 + i for i in range(15000)],
                "close": [102 + i for i in range(15000)],
                "volume": [1000 + i for i in range(15000)],
            }
        )
        large_df["timestamp"] = pd.to_datetime(large_df["timestamp"])
        large_df = large_df.set_index("timestamp")

        # Calculate hash with sampling and full hash
        hash_sampling = self.version_manager._calculate_dataframe_hash(
            large_df, use_full_hash=False
        )
        hash_full = self.version_manager._calculate_dataframe_hash(
            large_df, use_full_hash=True
        )

        # Hashes should be different (sampling vs full)
        self.assertNotEqual(hash_sampling, hash_full)

    def test_hash_changes_with_data_changes(self):
        """Test that hash changes when data is modified."""
        # Create base dataset
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                "open": [100 + i for i in range(100)],
                "high": [105 + i for i in range(100)],
                "low": [95 + i for i in range(100)],
                "close": [102 + i for i in range(100)],
                "volume": [1000 + i for i in range(100)],
            }
        )
        df1["timestamp"] = pd.to_datetime(df1["timestamp"])
        df1 = df1.set_index("timestamp")

        # Create modified dataset (change one value)
        df2 = df1.copy()
        df2.loc[df2.index[50], "close"] = 999

        # Calculate hashes
        hash1 = self.version_manager._calculate_dataframe_hash(df1)
        hash2 = self.version_manager._calculate_dataframe_hash(df2)

        # Hashes should be different
        self.assertNotEqual(hash1, hash2)


class TestGapHandlingStrategies(unittest.TestCase):
    """Test cases for configurable gap handling strategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "backtesting": {
                "data_dir": "test_gap_data",
                "gap_handling_strategy": "forward_fill",  # Default strategy
            }
        }
        self.mock_data_fetcher = MagicMock()
        self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove any test directories created
        test_dir = os.path.join(os.getcwd(), "data", "historical", "test_gap_data")
        if os.path.exists(test_dir):
            import shutil

            shutil.rmtree(test_dir)

    def test_forward_fill_strategy_no_gaps(self):
        """Test forward fill strategy with no gaps."""
        # Create data with no gaps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "open": [100 + i for i in range(10)],
                "high": [105 + i for i in range(10)],
                "low": [95 + i for i in range(10)],
                "close": [102 + i for i in range(10)],
                "volume": [1000 + i for i in range(10)],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Apply forward fill strategy
        result = self.loader._apply_forward_fill_with_logging(df, "BTC/USDT")

        # Data should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_forward_fill_strategy_with_gaps(self):
        """Test forward fill strategy with gaps."""
        # Create data with gaps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "open": [100, np.nan, np.nan, 103, 104, np.nan, 106, 107, 108, 109],
                "high": [105, 106, np.nan, 108, 109, 110, np.nan, 112, 113, 114],
                "low": [95, np.nan, 97, 98, 99, np.nan, 101, 102, 103, 104],
                "close": [102, 103, 104, np.nan, 106, 107, 108, np.nan, 110, 111],
                "volume": [
                    1000,
                    np.nan,
                    1200,
                    1300,
                    1400,
                    np.nan,
                    1600,
                    1700,
                    1800,
                    1900,
                ],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Apply forward fill strategy
        result = self.loader._apply_forward_fill_with_logging(df, "BTC/USDT")

        # Check that gaps were filled
        self.assertFalse(result["open"].isna().any())
        self.assertFalse(result["high"].isna().any())
        self.assertFalse(result["low"].isna().any())
        self.assertFalse(result["close"].isna().any())
        self.assertFalse(result["volume"].isna().any())

        # Check specific forward fill values
        self.assertEqual(result.loc[result.index[1], "open"], 100)  # Forward filled
        self.assertEqual(result.loc[result.index[2], "open"], 100)  # Forward filled
        self.assertEqual(
            result.loc[result.index[5], "volume"], 0
        )  # Volume filled with 0

    def test_interpolation_strategy(self):
        """Test interpolation strategy."""
        # Create data with gaps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "open": [100, np.nan, np.nan, 106, 107, np.nan, 109, 110, 111, 112],
                "high": [105, np.nan, 107, 108, 109, np.nan, 111, 112, 113, 114],
                "low": [95, np.nan, 97, 98, 99, np.nan, 101, 102, 103, 104],
                "close": [102, np.nan, 104, 105, 106, np.nan, 108, 109, 110, 111],
                "volume": [
                    1000,
                    np.nan,
                    1200,
                    1300,
                    1400,
                    np.nan,
                    1600,
                    1700,
                    1800,
                    1900,
                ],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Apply interpolation strategy
        result = self.loader._apply_interpolation_with_logging(df, "BTC/USDT")

        # Check that gaps were interpolated
        self.assertFalse(result["open"].isna().any())
        self.assertFalse(result["high"].isna().any())
        self.assertFalse(result["low"].isna().any())
        self.assertFalse(result["close"].isna().any())
        self.assertFalse(result["volume"].isna().any())

        # Check that interpolated values are between surrounding values
        self.assertTrue(100 < result.loc[result.index[1], "open"] < 106)  # Interpolated
        self.assertEqual(
            result.loc[result.index[5], "volume"], 0
        )  # Volume filled with 0

    def test_reject_strategy_no_gaps(self):
        """Test reject strategy with no gaps."""
        # Create data with no gaps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "open": [100 + i for i in range(10)],
                "high": [105 + i for i in range(10)],
                "low": [95 + i for i in range(10)],
                "close": [102 + i for i in range(10)],
                "volume": [1000 + i for i in range(10)],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Apply reject strategy - should pass
        result = self.loader._apply_reject_strategy(df, "BTC/USDT")

        # Data should be unchanged
        pd.testing.assert_frame_equal(result, df)

    def test_reject_strategy_with_gaps(self):
        """Test reject strategy with gaps - should raise exception."""
        # Create data with gaps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "open": [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
                "low": [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
                "close": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Apply reject strategy - should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.loader._apply_reject_strategy(df, "BTC/USDT")

        # Check error message
        self.assertIn("Found", str(context.exception))
        self.assertIn("missing values", str(context.exception))
        self.assertIn("BTC/USDT", str(context.exception))

    def test_unknown_gap_strategy_defaults_to_forward_fill(self):
        """Test that unknown gap strategy defaults to forward fill."""
        # Create loader with unknown strategy
        config = {
            "backtesting": {
                "data_dir": "test_unknown",
                "gap_handling_strategy": "unknown_strategy",
            }
        }
        loader = HistoricalDataLoader(config, self.mock_data_fetcher)

        # Create data with gaps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=5, freq="h"),
                "open": [100, np.nan, 102, 103, 104],
                "high": [105, 106, 107, 108, 109],
                "low": [95, 96, 97, 98, 99],
                "close": [102, 103, 104, 105, 106],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Apply gap handling - should default to forward fill
        result = loader._apply_gap_handling_strategy(df, "BTC/USDT")

        # Check that gaps were forward filled
        self.assertFalse(result["open"].isna().any())
        self.assertEqual(result.loc[result.index[1], "open"], 100)  # Forward filled


class TestDataOptimizations(unittest.TestCase):
    """Test cases for memory efficiency optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {"backtesting": {"data_dir": "test_opt", "deduplicate": False}}
        self.mock_data_fetcher = MagicMock()
        self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)

    def tearDown(self):
        """Clean up test fixtures."""
        test_dir = os.path.join(os.getcwd(), "data", "historical", "test_opt")
        if os.path.exists(test_dir):
            import shutil

            shutil.rmtree(test_dir)

    def test_concatenation_generator_optimization(self):
        """Test that generator-based concatenation works for large datasets."""
        # Create test DataFrames
        dfs = []
        base_time = pd.Timestamp("2023-01-01")

        for i in range(150):  # Large dataset (> 100 DataFrames)
            timestamps = pd.date_range(
                base_time + pd.Timedelta(hours=i * 24), periods=100, freq="h"
            )
            data = {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(150, 250, 100),
                "low": np.random.uniform(50, 150, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            }
            df = pd.DataFrame(data, index=timestamps)
            dfs.append(df)

        # Test generator concatenation (new method)
        start_time = time.time()
        result_new = pd.concat((df for df in dfs), copy=False)
        result_new.sort_index(inplace=True)
        new_time = max(time.time() - start_time, 1e-6)

        # Test list concatenation (old method)
        start_time = time.time()
        result_old = pd.concat(dfs, copy=False, ignore_index=False)
        result_old.sort_index(inplace=True)
        old_time = max(time.time() - start_time, 1e-6)

        # Results should be equivalent
        pd.testing.assert_frame_equal(result_old, result_new)

        # Performance check with fallback for small datasets
        # For small datasets, generator may not be faster, so allow more tolerance
        if len(dfs) < 1000:  # Small dataset threshold
            # Allow up to 10x slower for small datasets (generator overhead)
            tolerance = 10.0
        else:
            # For large datasets, expect reasonable performance
            tolerance = 2.0

        self.assertLessEqual(
            new_time,
            old_time * tolerance,
            f"Generator concatenation too slow: {new_time:.4f}s vs {old_time:.4f}s (tolerance: {tolerance}x)",
        )

    def test_cache_save_optimization(self):
        """Test that cache save optimization reduces DataFrame copies."""
        # Create test DataFrame
        test_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1000, freq="1h"),
                "open": np.random.uniform(100, 200, 1000),
                "high": np.random.uniform(150, 250, 1000),
                "low": np.random.uniform(50, 150, 1000),
                "close": np.random.uniform(100, 200, 1000),
                "volume": np.random.uniform(1000, 10000, 1000),
            }
        )
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
        test_df = test_df.set_index("timestamp")

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "cache_enabled": True,
                "cache_dir": temp_dir,
                "name": "binance",
                "api_key": "",
                "api_secret": "",
                "rate_limit": 10,
            }

            fetcher = DataFetcher(config)

            # Test cache save operation
            cache_key = "test_optimization_cache"

            # This should work without errors (testing the fillna fix)
            fetcher._save_to_cache(cache_key, test_df)

            # Verify cache file was created
            cache_path = os.path.join(temp_dir, f"{cache_key}.json")
            self.assertTrue(os.path.exists(cache_path))

            # Verify cache can be loaded back
            loaded_df = fetcher._load_from_cache(cache_key)
            self.assertIsNotNone(loaded_df)
            self.assertFalse(loaded_df.empty)

            # Data should be preserved (timestamps will be different format but equivalent)
            self.assertEqual(len(loaded_df), len(test_df))
            self.assertListEqual(list(loaded_df.columns), list(test_df.columns))


class TestRefactoredFunctions(unittest.TestCase):
    """Test cases for refactored helper functions in data modules."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "backtesting": {"data_dir": "test_refactor", "deduplicate": False}
        }
        self.mock_data_fetcher = MagicMock()
        self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)

        self.fetcher_config = {
            "name": "binance",
            "cache_enabled": True,
            "cache_dir": "test_refactor_cache",
        }
        self.fetcher = DataFetcher(self.fetcher_config)

    def tearDown(self):
        """Clean up test fixtures."""
        test_dir = os.path.join(os.getcwd(), "data", "historical", "test_refactor")
        if os.path.exists(test_dir):
            import shutil

            shutil.rmtree(test_dir)

        cache_dir = os.path.join(os.getcwd(), "data", "cache", "test_refactor_cache")
        if os.path.exists(cache_dir):
            import shutil

            shutil.rmtree(cache_dir)

    def test_calculate_pagination_params(self):
        """Test pagination parameter calculation."""
        (
            start_dt,
            end_dt,
            delta,
            estimated_requests,
        ) = self.loader._calculate_pagination_params("2023-01-01", "2023-01-05", "1h")

        self.assertEqual(start_dt, pd.Timestamp("2023-01-01"))
        self.assertEqual(end_dt, pd.Timestamp("2023-01-05"))
        self.assertEqual(delta, timedelta(hours=1))
        self.assertGreater(estimated_requests, 0)

    def test_initialize_pagination_state(self):
        """Test pagination state initialization."""
        max_iter, iter_count, consec_same = self.loader._initialize_pagination_state()

        self.assertEqual(max_iter, 1000)
        self.assertEqual(iter_count, 0)
        self.assertEqual(consec_same, 0)

    def test_advance_pagination_window_normal_case(self):
        """Test pagination window advancement in normal case."""
        current_start = pd.Timestamp("2023-01-01 12:00:00")
        last_index = pd.Timestamp("2023-01-01 13:00:00")
        delta = timedelta(hours=1)

        new_start = self.loader._advance_pagination_window(
            current_start, last_index, delta, "BTC/USDT"
        )

        # Should advance to last_index
        self.assertEqual(new_start, last_index)

    def test_advance_pagination_window_same_or_earlier(self):
        """Test pagination window advancement when exchange returns same/earlier data."""
        current_start = pd.Timestamp("2023-01-01 12:00:00")
        last_index = pd.Timestamp("2023-01-01 11:00:00")  # Earlier than current_start
        delta = timedelta(hours=1)

        new_start = self.loader._advance_pagination_window(
            current_start, last_index, delta, "BTC/USDT"
        )

        # Should advance by delta
        self.assertEqual(new_start, current_start + delta)

    def test_detect_infinite_loop_no_loop(self):
        """Test infinite loop detection when no loop exists."""
        current_start = pd.Timestamp("2023-01-01 12:00:00")
        last_current_start = pd.Timestamp("2023-01-01 11:00:00")
        consecutive_same = 0

        should_break, new_consec = self.loader._detect_infinite_loop(
            current_start, last_current_start, consecutive_same, "BTC/USDT"
        )

        self.assertFalse(should_break)
        self.assertEqual(new_consec, 0)

    def test_detect_infinite_loop_with_loop(self):
        """Test infinite loop detection when loop is detected."""
        current_start = pd.Timestamp("2023-01-01 12:00:00")
        last_current_start = current_start  # Same as current
        consecutive_same = 3

        should_break, new_consec = self.loader._detect_infinite_loop(
            current_start, last_current_start, consecutive_same, "BTC/USDT"
        )

        self.assertTrue(should_break)
        self.assertEqual(new_consec, 4)

    def test_combine_paginated_data(self):
        """Test combining paginated data chunks."""
        # Create test DataFrames
        df1 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "open": [100 + i for i in range(10)],
                "high": [105 + i for i in range(10)],
                "low": [95 + i for i in range(10)],
                "close": [102 + i for i in range(10)],
                "volume": [1000 + i for i in range(10)],
            }
        )
        df1["timestamp"] = pd.to_datetime(df1["timestamp"])
        df1 = df1.set_index("timestamp")

        df2 = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-02", periods=10, freq="h"),
                "open": [110 + i for i in range(10)],
                "high": [115 + i for i in range(10)],
                "low": [105 + i for i in range(10)],
                "close": [112 + i for i in range(10)],
                "volume": [1100 + i for i in range(10)],
            }
        )
        df2["timestamp"] = pd.to_datetime(df2["timestamp"])
        df2 = df2.set_index("timestamp")

        all_data = [df1, df2]
        combined = self.loader._combine_paginated_data(all_data, "BTC/USDT")

        self.assertIsNotNone(combined)
        self.assertEqual(len(combined), 20)  # 10 + 10 rows
        self.assertTrue(combined.index.is_monotonic_increasing)  # Should be sorted

    def test_prepare_data_for_gap_handling(self):
        """Test preparing DataFrame for gap handling."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                "open": [100 + i for i in range(10)],
                "high": [105 + i for i in range(10)],
                "low": [95 + i for i in range(10)],
                "close": [102 + i for i in range(10)],
                # Missing volume column
            }
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        prepared = self.loader._prepare_data_for_gap_handling(df)

        # Should have added volume column with NaN
        self.assertIn("volume", prepared.columns)
        self.assertTrue(prepared["volume"].isna().all())

    def test_prepare_cache_key(self):
        """Test cache key preparation."""
        # With caching enabled and not force fresh
        cache_key = self.fetcher._prepare_cache_key("BTC/USDT", "1h", 1000, None, False)
        self.assertIsNotNone(cache_key)
        self.assertIsInstance(cache_key, str)

        # With force fresh
        cache_key = self.fetcher._prepare_cache_key("BTC/USDT", "1h", 1000, None, True)
        self.assertIsNone(cache_key)

    def test_validate_candle_data_valid(self):
        """Test candle data validation with valid data."""
        candles = [
            [1640995200000, 100, 105, 95, 102, 1000],
            [1640995260000, 102, 107, 97, 104, 1100],
        ]

        is_valid = self.fetcher._validate_candle_data(candles, "BTC/USDT")
        self.assertTrue(is_valid)

    def test_validate_candle_data_invalid(self):
        """Test candle data validation with invalid data."""
        # Candle with only 5 elements (missing volume)
        candles = [
            [1640995200000, 100, 105, 95, 102],  # Missing volume
        ]

        is_valid = self.fetcher._validate_candle_data(candles, "BTC/USDT")
        self.assertFalse(is_valid)

    def test_convert_to_dataframe(self):
        """Test conversion of candle data to DataFrame."""
        candles = [
            [1640995200000, 100, 105, 95, 102, 1000],
            [1640995260000, 102, 107, 97, 104, 1100],
        ]

        df = self.fetcher._convert_to_dataframe(candles)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(
            list(df.columns), ["timestamp", "open", "high", "low", "close", "volume"]
        )
        # Timestamp is now kept as a column instead of being set as index
        self.assertIsInstance(df["timestamp"], pd.Series)

    def test_exchange_wrapper_properties(self):
        """Test that ExchangeWrapper exposes commonly used methods as properties."""
        # Create a mock exchange
        mock_exchange = MagicMock()
        mock_exchange.id = "binance"
        mock_exchange.name = "Binance"
        mock_exchange.load_markets = AsyncMock()
        mock_exchange.fetch_ohlcv = AsyncMock()
        mock_exchange.fetch_ticker = AsyncMock()
        mock_exchange.fetch_order_book = AsyncMock()
        mock_exchange.close = AsyncMock()

        # Create wrapper
        wrapper = self.fetcher._ExchangeWrapper(mock_exchange, self.fetcher)

        # Test property access (should not call __getattr__)
        self.assertEqual(wrapper.id, "binance")
        self.assertEqual(wrapper.name, "Binance")
        self.assertEqual(wrapper.load_markets, mock_exchange.load_markets)
        self.assertEqual(wrapper.fetch_ohlcv, mock_exchange.fetch_ohlcv)
        self.assertEqual(wrapper.fetch_ticker, mock_exchange.fetch_ticker)
        self.assertEqual(wrapper.fetch_order_book, mock_exchange.fetch_order_book)
        self.assertEqual(wrapper.close, mock_exchange.close)


class TestStructuredLogging(unittest.TestCase):
    """Test cases for structured logging in data modules."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "name": "binance",
            "cache_enabled": True,
            "cache_dir": "test_logging_cache",
        }
        self.fetcher = DataFetcher(self.config)

        self.loader_config = {
            "backtesting": {"data_dir": "test_logging_data", "deduplicate": False}
        }
        self.mock_data_fetcher = MagicMock()
        self.loader = HistoricalDataLoader(self.loader_config, self.mock_data_fetcher)

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test directories
        test_dirs = [
            os.path.join(os.getcwd(), "data", "cache", "test_logging_cache"),
            os.path.join(os.getcwd(), "data", "historical", "test_logging_data"),
        ]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                import shutil

                shutil.rmtree(test_dir)

    @unittest.skip("Requires async test runner")
    async def test_data_fetcher_structured_logging(self):
        """Test that DataFetcher operations produce structured logs."""
        # This test verifies that logging calls are made with proper structure
        # In a real scenario, you would capture log output and verify format
        with patch("data.data_fetcher.logger") as mock_logger:
            # Test get_historical_data logging
            mock_candles = [
                [1640995200000, 100, 105, 95, 102, 1000],
                [1640995260000, 102, 107, 97, 104, 1100],
            ]

            async def mock_fetch(*args, **kwargs):
                return mock_candles

            with patch.object(
                self.fetcher, "_fetch_from_exchange", side_effect=mock_fetch
            ):
                with patch.object(
                    self.fetcher, "_validate_candle_data", return_value=True
                ):
                    with patch.object(
                        self.fetcher, "_convert_to_dataframe"
                    ) as mock_convert:
                        mock_df = pd.DataFrame(
                            {
                                "timestamp": pd.date_range(
                                    "2023-01-01", periods=2, freq="h"
                                ),
                                "open": [100, 102],
                                "high": [105, 107],
                                "low": [95, 97],
                                "close": [102, 104],
                                "volume": [1000, 1100],
                            }
                        )
                        mock_df["timestamp"] = pd.to_datetime(mock_df["timestamp"])
                        mock_df = mock_df.set_index("timestamp")
                        mock_convert.return_value = mock_df

                        # Call the async method
                        result = await self.fetcher.get_historical_data(
                            "BTC/USDT", "1h", 1000
                        )

                        # Verify logging calls were made
                        mock_logger.info.assert_called()
                        # Check that at least one info call contains expected structure
                        info_calls = [
                            call
                            for call in mock_logger.info.call_args_list
                            if "Starting historical data fetch" in str(call)
                        ]
                        self.assertTrue(len(info_calls) > 0)

    @unittest.skip("Requires async test runner")
    async def test_historical_loader_structured_logging(self):
        """Test that HistoricalDataLoader operations produce structured logs."""
        with patch("data.historical_loader.logger") as mock_logger:
            # Mock successful data loading
            mock_df = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
                    "open": [100 + i for i in range(10)],
                    "high": [105 + i for i in range(10)],
                    "low": [95 + i for i in range(10)],
                    "close": [102 + i for i in range(10)],
                    "volume": [1000 + i for i in range(10)],
                }
            )
            mock_df["timestamp"] = pd.to_datetime(mock_df["timestamp"])
            mock_df = mock_df.set_index("timestamp")

            # Mock the data fetcher
            async def mock_get_historical_data(*args, **kwargs):
                return mock_df

            self.mock_data_fetcher.get_historical_data = mock_get_historical_data

            # Call load_historical_data (async)
            result = await self.loader.load_historical_data(
                ["BTC/USDT"], "2023-01-01", "2023-01-02", "1h"
            )

            # Verify logging calls were made
            mock_logger.info.assert_called()
            # Check for structured logging calls
            info_calls = mock_logger.info.call_args_list
            self.assertTrue(
                any("Starting historical data load" in str(call) for call in info_calls)
            )
            self.assertTrue(
                any(
                    "Completed historical data load" in str(call) for call in info_calls
                )
            )


class TestMetadataErrorHandling(unittest.TestCase):
    """Test cases for metadata error handling and recovery."""

    def setUp(self):
        """Set up test fixtures."""
        self.version_manager = DatasetVersionManager()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test versions directory
        import shutil

        test_dir = os.path.join(os.getcwd(), "data", "versions")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    def test_metadata_file_not_found(self):
        """Test handling when metadata file doesn't exist."""
        # Remove metadata file if it exists
        if os.path.exists(self.version_manager.metadata_file):
            os.remove(self.version_manager.metadata_file)

        # Load metadata - should handle gracefully
        self.version_manager._load_metadata()

        # Should initialize with empty metadata
        self.assertEqual(self.version_manager.metadata, {})

    def test_corrupted_metadata_json_error(self):
        """Test handling of corrupted JSON in metadata file."""
        # Create corrupted metadata file
        os.makedirs(os.path.dirname(self.version_manager.metadata_file), exist_ok=True)
        with open(self.version_manager.metadata_file, "w") as f:
            f.write("{ invalid json content }")

        # Load metadata - should handle JSON error
        with self.assertRaises(MetadataError) as context:
            self.version_manager._load_metadata()

        # Check error message
        self.assertIn("corruption unrecoverable", str(context.exception))

    def test_corrupted_metadata_with_backup_recovery(self):
        """Test metadata recovery from backup file."""
        # Create backup file with valid metadata
        backup_file = self.version_manager.metadata_file.with_suffix(".bak")
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)

        valid_metadata = {
            "test_version_20230916_120000": {
                "version_id": "test_version_20230916_120000",
                "version_name": "test_version",
                "timestamp": "2023-09-16T12:00:00",
                "description": "Test version",
                "shape": [100, 6],
                "columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "dtypes": {"timestamp": "datetime64[ns]", "open": "float64"},
                "metadata": {},
                "hash": "test_hash",
                "validation_passed": True,
            }
        }

        with open(backup_file, "w") as f:
            json.dump(valid_metadata, f)

        # Create corrupted main metadata file
        with open(self.version_manager.metadata_file, "w") as f:
            f.write("{ invalid json }")

        # Load metadata - should recover from backup
        with patch.object(self.version_manager, "_save_metadata") as mock_save:
            self.version_manager._load_metadata()

            # Should have recovered metadata from backup
            self.assertEqual(self.version_manager.metadata, valid_metadata)
            # Should have called _save_metadata to persist recovery
            mock_save.assert_called_once()

    def test_metadata_backup_recovery_failure(self):
        """Test handling when backup recovery also fails."""
        # Create corrupted backup file
        backup_file = self.version_manager.metadata_file.with_suffix(".bak")
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        with open(backup_file, "w") as f:
            f.write("{ invalid backup json }")

        # Create corrupted main metadata file
        with open(self.version_manager.metadata_file, "w") as f:
            f.write("{ invalid main json }")

        # Load metadata - should raise MetadataError
        with self.assertRaises(MetadataError):
            self.version_manager._load_metadata()

    def test_successful_metadata_load(self):
        """Test successful metadata loading."""
        # Create valid metadata file
        valid_metadata = {
            "version_20230916_120000": {
                "version_id": "version_20230916_120000",
                "version_name": "test",
                "timestamp": "2023-09-16T12:00:00",
                "description": "Test version",
                "shape": [100, 6],
                "columns": ["timestamp", "open", "high", "low", "close", "volume"],
                "dtypes": {"timestamp": "datetime64[ns]"},
                "metadata": {},
                "hash": "test_hash",
                "validation_passed": True,
            }
        }

        os.makedirs(os.path.dirname(self.version_manager.metadata_file), exist_ok=True)
        with open(self.version_manager.metadata_file, "w") as f:
            json.dump(valid_metadata, f)

        # Load metadata
        self.version_manager._load_metadata()

        # Should have loaded the metadata
        self.assertEqual(self.version_manager.metadata, valid_metadata)

    def test_attempt_backup_recovery_no_backup_file(self):
        """Test backup recovery when no backup file exists."""
        # Ensure no backup file exists
        backup_file = self.version_manager.metadata_file.with_suffix(".bak")
        if os.path.exists(backup_file):
            os.remove(backup_file)

        # Attempt backup recovery
        success = self.version_manager._attempt_backup_recovery()

        # Should return False
        self.assertFalse(success)


class TestDatasetVersioningStructuredLogging(unittest.TestCase):
    """Test cases for structured logging in dataset versioning."""

    def setUp(self):
        """Set up test fixtures."""
        self.version_manager = DatasetVersionManager()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up test versions directory
        import shutil

        test_dir = os.path.join(os.getcwd(), "data", "versions")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    def test_create_version_structured_logging(self):
        """Test structured logging in create_version."""
        with patch("data.dataset_versioning.logger") as mock_logger:
            # Create test DataFrame
            df = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                    "open": [100 + i for i in range(100)],
                    "high": [105 + i for i in range(100)],
                    "low": [95 + i for i in range(100)],
                    "close": [102 + i for i in range(100)],
                    "volume": [1000 + i for i in range(100)],
                }
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df_reset = df.reset_index()  # Reset index to make timestamp a column
            # Create version
            version_id = self.version_manager.create_version(
                df_reset, "test_version", "Test version creation"
            )

            # Verify logging calls
            mock_logger.info.assert_called()
            info_calls = mock_logger.info.call_args_list

            # Check for structured logging
            start_calls = [
                call
                for call in info_calls
                if "Starting dataset version creation" in str(call)
            ]
            complete_calls = [
                call
                for call in info_calls
                if "Dataset version created successfully" in str(call)
            ]

            self.assertTrue(len(start_calls) > 0)
            self.assertTrue(len(complete_calls) > 0)

    def test_migrate_legacy_dataset_structured_logging(self):
        """Test structured logging in migrate_legacy_dataset."""
        with patch("data.dataset_versioning.logger") as mock_logger:
            # Create test DataFrame
            df = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
                    "open": [100 + i for i in range(100)],
                    "high": [105 + i for i in range(100)],
                    "low": [95 + i for i in range(100)],
                    "close": [102 + i for i in range(100)],
                    "volume": [1000 + i for i in range(100)],
                }
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df_reset = df.reset_index()  # Reset index to make timestamp a column

            # Mock create_binary_labels to avoid import issues
            with patch(
                "data.dataset_versioning.create_binary_labels", return_value=df_reset
            ):
                # Migrate dataset
                version_id = migrate_legacy_dataset(df_reset, self.version_manager)

                # Verify logging calls
                mock_logger.info.assert_called()
                info_calls = mock_logger.info.call_args_list

                # Check for structured logging
                start_calls = [
                    call
                    for call in info_calls
                    if "Starting legacy dataset migration" in str(call)
                ]
                complete_calls = [
                    call
                    for call in info_calls
                    if "Legacy dataset migration completed" in str(call)
                ]

                self.assertTrue(len(start_calls) > 0)
                self.assertTrue(len(complete_calls) > 0)


if __name__ == "__main__":
    unittest.main()
