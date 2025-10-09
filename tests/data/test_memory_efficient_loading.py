"""
Memory-efficient loading tests for data/historical_loader.py

Tests streaming data processing, chunked loading, memory monitoring,
and large dataset handling with timeout protection.
"""

import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.memory_manager import get_memory_manager
from data.data_fetcher import DataFetcher
from data.historical_loader import HistoricalDataLoader


class TestMemoryEfficientLoading:
    """Test cases for memory-efficient data loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "backtesting": {
                "data_dir": "test_historical_data",
                "deduplicate": True,
                "chunk_size_mb": 50,
                "max_memory_mb": 200,
            }
        }
        self.mock_data_fetcher = MagicMock(spec=DataFetcher)
        self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)

    @pytest.mark.timeout(30)
    def test_large_dataset_loading_with_timeout(self):
        """Test loading large dataset with timeout protection."""
        # Generate large dataset that would normally cause memory issues
        large_dataset = self._generate_large_dataset(
            size_gb=0.1
        )  # Further reduced for testing

        # Process with timeout protection
        with ThreadPoolExecutor(max_workers=2) as executor:
            future = executor.submit(
                lambda: list(self.loader.load_chunked(large_dataset, chunk_size=1000))
            )
            try:
                chunks = future.result(timeout=25)  # Should complete within timeout
                assert len(chunks) > 0
                # Verify data integrity
                total_rows = sum(len(chunk) for chunk in chunks)
                assert total_rows == len(large_dataset)
            except TimeoutError:
                pytest.fail("Large dataset loading timed out")

    def test_chunked_loading_basic(self):
        """Test basic chunked loading functionality."""
        # Create test data
        data = self._generate_test_data(1000)

        chunks = list(self.loader.load_chunked(data, chunk_size=100))

        assert len(chunks) == 10  # 1000 rows / 100 chunk_size
        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)
            assert len(chunk) <= 100
            assert not chunk.empty

        # Verify data integrity - all chunks should concatenate to original
        reconstructed = pd.concat(chunks)
        reconstructed = reconstructed[~reconstructed.index.duplicated(keep="first")]
        pd.testing.assert_frame_equal(reconstructed.sort_index(), data.sort_index())

    def test_chunked_loading_with_memory_limits(self):
        """Test chunked loading respects memory limits."""
        data = self._generate_test_data(1000)

        # Set small chunk size to test memory limits
        chunks = list(self.loader.load_chunked(data, chunk_size=50, max_memory_mb=10))

        assert len(chunks) > 1
        for chunk in chunks:
            # Estimate memory usage of chunk
            memory_usage = chunk.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            assert memory_usage <= 15  # Allow some overhead

    def test_memory_monitoring_during_loading(self):
        """Test memory usage monitoring during chunked loading."""
        data = self._generate_test_data(500)

        memory_stats = []

        def monitor_memory():
            memory_manager = get_memory_manager()
            stats = memory_manager.get_memory_stats()
            memory_stats.append(stats["current_memory_mb"])
            return stats

        # Load with monitoring
        chunks = list(
            self.loader.load_chunked(
                data, chunk_size=100, memory_monitor=monitor_memory
            )
        )

        assert len(chunks) > 0
        assert len(memory_stats) > 0

        # Memory usage should not grow unbounded
        max_memory = max(memory_stats)
        min_memory = min(memory_stats)
        assert max_memory - min_memory < 50  # Memory growth should be controlled

    def test_corrupted_chunk_handling(self):
        """Test graceful handling of corrupted data chunks."""
        # Create data with some corrupted chunks
        data = self._generate_test_data(300)

        # Introduce corruption in the middle
        data.iloc[100:150] = np.nan

        chunks = list(
            self.loader.load_chunked(data, chunk_size=50, handle_corruption=True)
        )

        # Should still process chunks, but corrupted ones might be filtered
        assert len(chunks) >= 4  # At least some chunks should be processed

        # Verify non-corrupted chunks are valid
        for chunk in chunks:
            if not chunk.empty:
                assert not chunk.isnull().all().all()  # Not completely corrupted

    def test_resume_capability(self):
        """Test resume capability for interrupted loads."""
        data = self._generate_test_data(1000)

        # Simulate partial load
        resume_state = {"last_index": 500, "processed_chunks": 5}

        # Resume loading from checkpoint
        remaining_chunks = list(
            self.loader.load_chunked(data, chunk_size=100, resume_from=resume_state)
        )

        assert len(remaining_chunks) >= 5  # Should load remaining data

        # Verify continuity
        all_chunks = []
        all_chunks.extend([pd.DataFrame()] * 5)  # Mock previous chunks
        all_chunks.extend(remaining_chunks)

        total_rows = sum(len(chunk) for chunk in remaining_chunks)
        assert total_rows >= 500  # At least the remaining data

    def test_progress_tracking(self):
        """Test progress tracking during chunked loading."""
        data = self._generate_test_data(1000)

        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress)

        chunks = list(
            self.loader.load_chunked(
                data, chunk_size=200, progress_callback=progress_callback
            )
        )

        assert len(progress_updates) > 0
        assert progress_updates[-1] >= 100  # Should reach 100% completion

        # Progress should be monotonically increasing
        for i in range(1, len(progress_updates)):
            assert progress_updates[i] >= progress_updates[i - 1]

    def test_configurable_chunk_sizes(self):
        """Test configurable chunk sizes based on memory."""
        data = self._generate_test_data(1000)

        # Test different chunk sizes
        for chunk_size in [50, 100, 200]:
            chunks = list(self.loader.load_chunked(data, chunk_size=chunk_size))
            expected_chunks = (len(data) + chunk_size - 1) // chunk_size
            assert len(chunks) == expected_chunks

    def test_memory_manager_integration(self):
        """Test integration with memory manager."""
        memory_manager = get_memory_manager()

        # Mock memory manager methods
        memory_manager.get_memory_stats = MagicMock(
            return_value={
                "current_memory_mb": 100,
                "warning_mb": 150,
                "critical_mb": 200,
            }
        )

        data = self._generate_test_data(500)

        # Load with memory manager integration
        chunks = list(
            self.loader.load_chunked(
                data, chunk_size=100, memory_manager=memory_manager
            )
        )

        assert len(chunks) > 0
        memory_manager.get_memory_stats.assert_called()

    def test_data_integrity_preservation(self):
        """Test that data integrity and ordering is preserved."""
        # Create data with specific ordering
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=1000, freq="1min"),
                "value": range(1000),
            }
        ).set_index("timestamp")

        chunks = list(self.loader.load_chunked(data, chunk_size=100))

        # Reconstruct data
        reconstructed = pd.concat(chunks)

        # Verify ordering is preserved
        assert reconstructed.index.is_monotonic_increasing

        # Verify all data is present
        assert len(reconstructed) == len(data)

        # Verify values are correct
        pd.testing.assert_series_equal(
            reconstructed["value"].sort_index(), data["value"].sort_index()
        )

    def _generate_test_data(self, size):
        """Generate test data of specified size."""
        return pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, size),
                "high": np.random.uniform(200, 300, size),
                "low": np.random.uniform(50, 100, size),
                "close": np.random.uniform(100, 200, size),
                "volume": np.random.uniform(1000, 10000, size),
            },
            index=pd.date_range("2023-01-01", periods=size, freq="1min"),
        )

    def _generate_large_dataset(self, size_gb):
        """Generate a large dataset for memory testing."""
        # Estimate rows needed for target size (rough approximation)
        rows_per_gb = 50000  # Adjust based on actual data size
        total_rows = int(size_gb * rows_per_gb)

        return self._generate_test_data(total_rows)


class TestMemoryEfficientAsyncLoading:
    """Test cases for async memory-efficient loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "backtesting": {
                "data_dir": "test_historical_data",
                "chunk_size_mb": 50,
                "max_memory_mb": 200,
            }
        }
        self.mock_data_fetcher = MagicMock(spec=DataFetcher)
        self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_async_chunked_loading(self):
        """Test async chunked loading with timeout."""
        # Generate test data
        data = self._generate_test_data(1000)

        # Load asynchronously with chunking
        chunks = []
        async for chunk in self.loader.load_chunked_async(data, chunk_size=100):
            chunks.append(chunk)

        assert len(chunks) == 10
        for chunk in chunks:
            assert isinstance(chunk, pd.DataFrame)
            assert len(chunk) <= 100

    @pytest.mark.asyncio
    async def test_async_memory_monitoring(self):
        """Test memory monitoring in async loading."""
        data = self._generate_test_data(500)

        memory_readings = []

        async def memory_monitor():
            # Simulate memory monitoring
            memory_readings.append(time.time())
            return {"memory_mb": 100}

        chunks = []
        async for chunk in self.loader.load_chunked_async(
            data, chunk_size=50, memory_monitor=memory_monitor
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert len(memory_readings) > 0

    def _generate_test_data(self, size):
        """Generate test data."""
        return pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, size),
                "high": np.random.uniform(200, 300, size),
                "low": np.random.uniform(50, 100, size),
                "close": np.random.uniform(100, 200, size),
                "volume": np.random.uniform(1000, 10000, size),
            },
            index=pd.date_range("2023-01-01", periods=size, freq="1min"),
        )


class TestMemoryEfficientHistoricalLoaderIntegration:
    """Integration tests for memory-efficient historical loading."""

    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            self.config = {
                "backtesting": {
                    "data_dir": self.temp_dir,
                    "chunk_size_mb": 50,
                    "max_memory_mb": 200,
                }
            }
            self.mock_data_fetcher = MagicMock(spec=DataFetcher)
            self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)

    @pytest.mark.asyncio
    async def test_full_memory_efficient_workflow(self):
        """Test complete memory-efficient loading workflow."""
        # Mock large dataset response
        large_data = self._generate_large_historical_data(2000)

        self.mock_data_fetcher.get_historical_data = AsyncMock(return_value=large_data)

        # Load with memory-efficient chunking
        result = await self.loader.load_historical_data_chunked(
            ["BTC/USDT"], "2023-01-01", "2023-01-10", "1h", chunk_size=500
        )

        assert "BTC/USDT" in result
        assert len(result["BTC/USDT"]) == 2000

        # Verify data integrity
        assert not result["BTC/USDT"].empty
        assert result["BTC/USDT"].index.is_monotonic_increasing

    @pytest.mark.asyncio
    async def test_memory_efficient_resampling(self):
        """Test memory-efficient resampling."""
        data = {"BTC/USDT": self._generate_large_historical_data(1000)}

        # Resample with memory efficiency
        resampled = await self.loader.resample_data_chunked(data, "4h", chunk_size=200)

        assert "BTC/USDT" in resampled
        # Should have fewer rows after resampling
        assert len(resampled["BTC/USDT"]) < len(data["BTC/USDT"])

    def _generate_large_historical_data(self, size):
        """Generate large historical OHLCV data."""
        return pd.DataFrame(
            {
                "open": np.random.uniform(40000, 60000, size),
                "high": np.random.uniform(40000, 65000, size),
                "low": np.random.uniform(35000, 40000, size),
                "close": np.random.uniform(40000, 60000, size),
                "volume": np.random.uniform(100, 1000, size),
            },
            index=pd.date_range("2023-01-01", periods=size, freq="1h"),
        )
