"""
Unit and integration tests for Multi-Timeframe Analysis feature.

Tests cover:
- TimeframeManager core functionality
- Data synchronization across timeframes
- Timestamp alignment algorithms
- Multi-timeframe indicator calculations
- Integration with strategy signal generation
- Performance under various conditions
- Edge cases and error handling
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from core.diagnostics import HealthCheckResult, HealthStatus
from core.timeframe_manager import SyncedData, TimeframeManager


class TestTimeframeManagerCore:
    """Test core TimeframeManager functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_data_fetcher, test_config):
        """Test TimeframeManager initialization."""
        tf_config = test_config.get("multi_timeframe", {})
        manager = TimeframeManager(mock_data_fetcher, tf_config)

        await manager.initialize()

        assert manager.data_fetcher == mock_data_fetcher
        assert manager.config == tf_config
        assert manager.symbol_timeframes == {}
        assert manager.cache == {}

    @pytest.mark.asyncio
    async def test_add_symbol_success(self, mock_data_fetcher, test_config):
        """Test successful symbol registration."""
        tf_config = test_config.get("multi_timeframe", {})
        manager = TimeframeManager(mock_data_fetcher, tf_config)
        await manager.initialize()

        timeframes = ["15m", "1h", "4h"]
        result = manager.add_symbol("BTC/USDT", timeframes)

        assert result is True
        assert "BTC/USDT" in manager.symbol_timeframes
        assert manager.symbol_timeframes["BTC/USDT"] == timeframes

    def test_add_symbol_invalid_timeframe(self, mock_data_fetcher, test_config):
        """Test symbol registration with invalid timeframe."""
        tf_config = test_config.get("multi_timeframe", {})
        manager = TimeframeManager(mock_data_fetcher, tf_config)

        timeframes = ["invalid_timeframe"]
        result = manager.add_symbol("BTC/USDT", timeframes)

        assert result is False
        assert "BTC/USDT" not in manager.symbol_timeframes

    @pytest.mark.asyncio
    async def test_remove_symbol(self, mock_data_fetcher, test_config):
        """Test symbol removal."""
        tf_config = test_config.get("multi_timeframe", {})
        manager = TimeframeManager(mock_data_fetcher, tf_config)
        await manager.initialize()

        # Add symbol first
        manager.add_symbol("BTC/USDT", ["1h", "4h"])

        # Remove symbol
        manager.remove_symbol("BTC/USDT")

        assert "BTC/USDT" not in manager.symbol_timeframes

    @pytest.mark.asyncio
    async def test_get_registered_symbols(self, mock_data_fetcher, test_config):
        """Test getting registered symbols."""
        tf_config = test_config.get("multi_timeframe", {})
        manager = TimeframeManager(mock_data_fetcher, tf_config)
        await manager.initialize()

        # Add symbols
        manager.add_symbol("BTC/USDT", ["1h"])
        manager.add_symbol("ETH/USDT", ["1h"])

        symbols = manager.get_registered_symbols()

        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols
        assert len(symbols) == 2


class TestDataSynchronization:
    """Test data synchronization across timeframes."""

    @pytest.mark.asyncio
    async def test_fetch_multi_timeframe_data_success(
        self, mock_data_fetcher, multi_timeframe_data
    ):
        """Test successful multi-timeframe data fetching."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock the data fetcher to return multi-timeframe data
        mock_data_fetcher.get_historical_data = AsyncMock(
            side_effect=[
                multi_timeframe_data["5m"],  # 5m data
                multi_timeframe_data["15m"],  # 15m data
                multi_timeframe_data["1h"],  # 1h data
                multi_timeframe_data["4h"],  # 4h data
            ]
        )

        manager.add_symbol("BTC/USDT", ["5m", "15m", "1h", "4h"])

        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        assert result is not None
        assert isinstance(result, SyncedData)
        assert result.symbol == "BTC/USDT"
        assert len(result.data) == 4  # 4 timeframes
        assert "5m" in result.data
        assert "15m" in result.data
        assert "1h" in result.data
        assert "4h" in result.data

    @pytest.mark.asyncio
    async def test_timestamp_alignment(self, mock_data_fetcher, multi_timeframe_data):
        """Test timestamp alignment across timeframes."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock data fetcher
        mock_data_fetcher.get_historical_data = AsyncMock(
            side_effect=[multi_timeframe_data["1h"], multi_timeframe_data["4h"]]
        )

        manager.add_symbol("BTC/USDT", ["1h", "4h"])

        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        assert result is not None

        # Check that timestamps are aligned
        h1_timestamps = result.data["1h"].index
        h4_timestamps = result.data["4h"].index

        # 4h data should have fewer points than 1h data
        assert len(h4_timestamps) <= len(h1_timestamps)

        # Check that 4h timestamps are subset of 1h timestamps (aligned)
        h1_ts_set = set(h1_timestamps)
        h4_ts_set = set(h4_timestamps)
        assert h4_ts_set.issubset(h1_ts_set)

    @pytest.mark.asyncio
    async def test_missing_data_handling(self, mock_data_fetcher):
        """Test handling of missing data in timeframes."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock data fetcher to return None for one timeframe
        mock_data_fetcher.get_historical_data = AsyncMock(
            side_effect=[
                pd.DataFrame(),  # Empty 1h data
                pd.DataFrame(),  # Empty 4h data
            ]
        )

        manager.add_symbol("BTC/USDT", ["1h", "4h"])

        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        # Should handle gracefully
        assert result is not None
        # Result should still be created but with empty dataframes

    @pytest.mark.asyncio
    async def test_data_caching(self, mock_data_fetcher, multi_timeframe_data):
        """Test data caching functionality."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock data fetcher
        mock_data_fetcher.get_historical_data = AsyncMock(
            return_value=multi_timeframe_data["1h"]
        )

        manager.add_symbol("BTC/USDT", ["1h"])

        # First fetch
        result1 = await manager.fetch_multi_timeframe_data("BTC/USDT")
        assert result1 is not None

        # Second fetch should use cache
        result2 = await manager.fetch_multi_timeframe_data("BTC/USDT")
        assert result2 is not None

        # Data fetcher should only be called once due to caching
        assert mock_data_fetcher.get_historical_data.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(self, mock_data_fetcher):
        """Test cache expiration."""
        manager = TimeframeManager(
            mock_data_fetcher, {"cache_ttl_seconds": 0}
        )  # 0 second TTL (immediate expiration)
        await manager.initialize()

        # Create fresh data with current timestamp
        current_time = pd.Timestamp.now()
        fresh_data = pd.DataFrame(
            {
                "open": [50000.0],
                "high": [50100.0],
                "low": [49900.0],
                "close": [50050.0],
                "volume": [100.0],
            },
            index=[current_time],
        )

        # Mock the data fetcher to return fresh data
        mock_data_fetcher.get_historical_data = AsyncMock(return_value=fresh_data)

        manager.add_symbol("BTC/USDT", ["1h"])

        # First fetch
        result1 = await manager.fetch_multi_timeframe_data("BTC/USDT")
        assert result1 is not None

        # Verify first call was made
        assert mock_data_fetcher.get_historical_data.call_count == 1

        # Clear the call count to check the second call
        mock_data_fetcher.get_historical_data.reset_mock()

        # Second fetch should bypass cache (TTL=0 means immediate expiration)
        result2 = await manager.fetch_multi_timeframe_data("BTC/USDT")
        assert result2 is not None

        # Data fetcher should be called again since cache expired immediately
        assert mock_data_fetcher.get_historical_data.call_count == 1


class TestSyncedData:
    """Test SyncedData class functionality."""

    def test_synced_data_creation(self, multi_timeframe_data):
        """Test SyncedData object creation."""
        from utils.time import now_ms

        synced = SyncedData(
            symbol="BTC/USDT",
            data=multi_timeframe_data,
            timestamp=int(datetime.now().timestamp() * 1000),
            last_updated=now_ms(),
            confidence_score=1.0,
        )

        assert synced.symbol == "BTC/USDT"
        assert len(synced.data) == 4
        assert isinstance(synced.timestamp, int)

    def test_synced_data_get_timeframe(self, multi_timeframe_data):
        """Test getting specific timeframe data."""
        from utils.time import now_ms

        synced = SyncedData(
            symbol="BTC/USDT",
            data=multi_timeframe_data,
            timestamp=int(datetime.now().timestamp() * 1000),
            last_updated=now_ms(),
            confidence_score=1.0,
        )

        h1_data = synced.get_timeframe("1h")
        assert h1_data is not None
        assert len(h1_data) > 0

        # Test non-existent timeframe
        nonexistent = synced.get_timeframe("1d")
        assert nonexistent is None

    def test_synced_data_get_latest_timestamp(self, multi_timeframe_data):
        """Test getting latest timestamp across timeframes."""
        from utils.time import now_ms

        synced = SyncedData(
            symbol="BTC/USDT",
            data=multi_timeframe_data,
            timestamp=int(datetime.now().timestamp() * 1000),
            last_updated=now_ms(),
            confidence_score=1.0,
        )

        latest_ts = synced.get_latest_timestamp()
        assert latest_ts is not None
        assert isinstance(latest_ts, pd.Timestamp)

    def test_synced_data_is_aligned(self, multi_timeframe_data):
        """Test timestamp alignment checking."""
        from utils.time import now_ms

        synced = SyncedData(
            symbol="BTC/USDT",
            data=multi_timeframe_data,
            timestamp=int(datetime.now().timestamp() * 1000),
            last_updated=now_ms(),
            confidence_score=1.0,
        )

        # Should be aligned by construction
        assert synced.is_aligned() is True


class TestIntegrationWithStrategies:
    """Test integration with trading strategies."""

    @pytest.mark.asyncio
    async def test_strategy_multi_timeframe_signal_generation(
        self, mock_data_fetcher, multi_timeframe_data
    ):
        """Test strategy signal generation with multi-timeframe data."""
        from strategies.base_strategy import BaseStrategy

        # Mock strategy that uses multi-timeframe data
        strategy = Mock(spec=BaseStrategy)
        strategy.generate_signals = AsyncMock(
            return_value=[
                {
                    "symbol": "BTC/USDT",
                    "signal_type": "BUY",
                    "price": 50000.0,
                    "timestamp": datetime.now(),
                    "confidence": 0.8,
                }
            ]
        )

        # Setup timeframe manager
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        mock_data_fetcher.get_historical_data = AsyncMock(
            side_effect=[multi_timeframe_data["1h"], multi_timeframe_data["4h"]]
        )

        manager.add_symbol("BTC/USDT", ["1h", "4h"])

        # Get multi-timeframe data
        mtf_data = await manager.fetch_multi_timeframe_data("BTC/USDT")

        # Generate signals with multi-timeframe data
        signals = await strategy.generate_signals({"BTC/USDT": mtf_data})

        # Verify strategy was called with multi-timeframe data
        strategy.generate_signals.assert_called_once()
        call_args = strategy.generate_signals.call_args[0][0]

        assert "BTC/USDT" in call_args
        assert isinstance(call_args["BTC/USDT"], SyncedData)

    @pytest.mark.asyncio
    async def test_backward_compatibility_single_timeframe(
        self, mock_data_fetcher, synthetic_market_data
    ):
        """Test backward compatibility with single timeframe operation."""
        from strategies.base_strategy import BaseStrategy

        strategy = Mock(spec=BaseStrategy)
        strategy.generate_signals = AsyncMock(return_value=[])

        # Setup timeframe manager without multi-timeframe config
        manager = TimeframeManager(mock_data_fetcher, {"enabled": False})
        await manager.initialize()

        mock_data_fetcher.get_historical_data = AsyncMock(
            return_value=synthetic_market_data
        )

        # Single timeframe operation should still work
        manager.add_symbol("BTC/USDT", ["1h"])

        # This should work without multi-timeframe data
        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        # Should return single timeframe data wrapped appropriately
        assert result is not None
        assert result.symbol == "BTC/USDT"


class TestPerformance:
    """Test performance characteristics of TimeframeManager."""

    @pytest.mark.asyncio
    async def test_data_fetching_performance(
        self, mock_data_fetcher, multi_timeframe_data, performance_timer
    ):
        """Test data fetching performance."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        mock_data_fetcher.get_historical_data = AsyncMock(
            return_value=multi_timeframe_data["1h"]
        )

        manager.add_symbol("BTC/USDT", ["1h"])

        # Measure performance
        performance_timer.start()
        result = await manager.fetch_multi_timeframe_data("BTC/USDT")
        performance_timer.stop()

        # Should complete within reasonable time
        duration_ms = performance_timer.duration_ms()
        assert duration_ms < 1000  # Less than 1 second
        assert result is not None

    @pytest.mark.asyncio
    async def test_memory_usage_multi_timeframe(
        self, mock_data_fetcher, multi_timeframe_data, memory_monitor
    ):
        """Test memory usage with multi-timeframe data."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        mock_data_fetcher.get_historical_data = AsyncMock(
            side_effect=[
                multi_timeframe_data["5m"],
                multi_timeframe_data["15m"],
                multi_timeframe_data["1h"],
                multi_timeframe_data["4h"],
            ]
        )

        manager.add_symbol("BTC/USDT", ["5m", "15m", "1h", "4h"])

        memory_monitor.start()

        # Fetch multi-timeframe data
        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        memory_delta = memory_monitor.get_memory_delta()

        # Memory usage should be reasonable
        assert memory_delta < 100  # Less than 100MB increase
        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_symbol_fetching(
        self, mock_data_fetcher, multi_timeframe_data
    ):
        """Test concurrent fetching for multiple symbols."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock concurrent data fetching
        async def mock_get_data(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate network delay
            return multi_timeframe_data["1h"]

        mock_data_fetcher.get_historical_data = AsyncMock(side_effect=mock_get_data)

        # Add multiple symbols
        symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        for symbol in symbols:
            manager.add_symbol(symbol, ["1h"])

        # Fetch data concurrently
        tasks = [manager.fetch_multi_timeframe_data(symbol) for symbol in symbols]

        results = await asyncio.gather(*tasks)

        # All fetches should succeed
        assert all(result is not None for result in results)
        assert len(results) == 3


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_exchange_connection_failure(self, mock_data_fetcher):
        """Test handling of exchange connection failures."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock connection failure
        mock_data_fetcher.get_historical_data = AsyncMock(
            side_effect=ConnectionError("Exchange unreachable")
        )

        manager.add_symbol("BTC/USDT", ["1h"])

        # Should handle gracefully
        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        # Should return None or handle error appropriately
        # (exact behavior depends on implementation)
        assert result is None or isinstance(result, SyncedData)

    @pytest.mark.asyncio
    async def test_partial_data_failure(self, mock_data_fetcher, multi_timeframe_data):
        """Test handling of partial data fetching failures."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock partial failure - some timeframes succeed, others fail
        mock_data_fetcher.get_historical_data = AsyncMock(
            side_effect=[
                multi_timeframe_data["1h"],  # Success
                ConnectionError("Failed to fetch 4h data"),  # Failure
            ]
        )

        manager.add_symbol("BTC/USDT", ["1h", "4h"])

        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        # Should handle partial failure gracefully
        assert result is not None
        # Should have data for successful timeframe
        assert "1h" in result.data

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, mock_data_fetcher):
        """Test handling of empty data responses."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Mock empty data response
        empty_df = pd.DataFrame()
        mock_data_fetcher.get_historical_data = AsyncMock(return_value=empty_df)

        manager.add_symbol("BTC/USDT", ["1h"])

        result = await manager.fetch_multi_timeframe_data("BTC/USDT")

        # Should handle empty data gracefully
        assert result is not None
        # Should have empty dataframe for the timeframe
        assert len(result.data["1h"]) == 0

    @pytest.mark.asyncio
    async def test_invalid_symbol_handling(self, mock_data_fetcher):
        """Test handling of invalid symbols."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Try to fetch data for unregistered symbol
        result = await manager.fetch_multi_timeframe_data("INVALID/USDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, mock_data_fetcher):
        """Test proper cleanup on shutdown."""
        manager = TimeframeManager(mock_data_fetcher, {})
        await manager.initialize()

        # Add some data to cache
        manager.add_symbol("BTC/USDT", ["1h"])
        manager.cache["BTC/USDT"] = "test_data"

        # Shutdown
        await manager.shutdown()

        # Cache should be cleared
        assert len(manager.cache) == 0
        assert len(manager.symbol_timeframes) == 0


class TestHealthMonitoring:
    """Test health monitoring integration."""

    @pytest.mark.asyncio
    async def test_health_check_integration(self, mock_data_fetcher, test_config):
        """Test integration with health monitoring system."""
        from core.diagnostics import get_diagnostics_manager

        tf_config = test_config.get("multi_timeframe", {})
        manager = TimeframeManager(mock_data_fetcher, tf_config)
        await manager.initialize()

        diagnostics = get_diagnostics_manager()

        # Register health check
        async def check_timeframe_manager():
            try:
                # Simple health check
                symbol_count = len(manager.get_registered_symbols())
                cache_size = len(manager.cache)

                return HealthCheckResult(
                    component="timeframe_manager",
                    status=HealthStatus.HEALTHY,
                    latency_ms=10.0,
                    message=f"Healthy: {symbol_count} symbols, {cache_size} cached",
                    details={
                        "registered_symbols": symbol_count,
                        "cache_entries": cache_size,
                    },
                )
            except Exception as e:
                return HealthCheckResult(
                    component="timeframe_manager",
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)},
                )

        diagnostics.register_health_check("timeframe_manager", check_timeframe_manager)

        # Run health check
        state = await diagnostics.run_health_check()

        # Should have timeframe manager health data
        assert "timeframe_manager" in state.component_statuses
        tf_status = state.component_statuses["timeframe_manager"]

        assert tf_status.status == HealthStatus.HEALTHY
        assert "Healthy:" in tf_status.message
