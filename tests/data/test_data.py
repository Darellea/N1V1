"""
tests/test_data.py

Unit tests for data fetching and historical data management.
Tests exchange connectivity, data validation, caching, and error handling.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import aiohttp
import os
from pathlib import Path

from data.data_fetcher import DataFetcher
from data.historical_loader import HistoricalDataLoader
from utils.config_loader import ConfigLoader


@pytest.fixture
def mock_exchange_config():
    """Fixture providing mock exchange configuration."""
    return {
        "name": "kucoin",
        "api_key": "test_key",
        "api_secret": "test_secret",
        "api_passphrase": "test_passphrase",
        "sandbox": False,
        "timeout": 30000,
        "rate_limit": 10,
        "cache_enabled": True,
        "cache_dir": ".test_cache",
    }


@pytest.fixture
def mock_ohlcv_data():
    """Fixture providing mock OHLCV data."""
    return [
        [
            1609459200000,
            100.0,
            102.0,
            98.0,
            101.0,
            1000.0,
        ],  # [timestamp, o, h, l, c, v]
        [1609545600000, 101.0, 103.0, 99.0, 102.0, 1200.0],
        [1609632000000, 102.0, 104.0, 100.0, 101.0, 800.0],
    ]


@pytest_asyncio.fixture
async def data_fetcher(mock_exchange_config):
    """Fixture providing initialized DataFetcher instance."""
    fetcher = DataFetcher(mock_exchange_config)
    fetcher.exchange = AsyncMock()  # Mock CCXT exchange
    fetcher.session = (
        aiohttp.ClientSession()
    )  # Real session but won't make actual calls
    yield fetcher
    await fetcher.shutdown()


@pytest.fixture
def historical_loader(mock_exchange_config):
    """Fixture providing HistoricalDataLoader instance."""
    mock_fetcher = AsyncMock(spec=DataFetcher)
    return HistoricalDataLoader(
        {"backtesting": {"data_dir": ".test_historical_data", "commission": 0.001}},
        mock_fetcher,
    )


@pytest.mark.asyncio
async def test_data_fetcher_initialization(data_fetcher, mock_exchange_config):
    """Test DataFetcher initialization and exchange connection."""
    assert data_fetcher.exchange is not None
    assert data_fetcher.config["name"] == mock_exchange_config["name"]
    assert data_fetcher.cache_enabled == mock_exchange_config["cache_enabled"]

    # Test markets loaded
    await data_fetcher.initialize()
    assert data_fetcher.exchange.load_markets.called


@pytest.mark.asyncio
async def test_fetch_ohlcv_data(data_fetcher, mock_ohlcv_data):
    """Test OHLCV data fetching from exchange."""
    # Mock the exchange fetch_ohlcv method directly
    data_fetcher.exchange._exchange.fetch_ohlcv = AsyncMock(
        return_value=mock_ohlcv_data
    )

    # Test successful fetch
    df = await data_fetcher.get_historical_data("BTC/USDT", "1h", limit=3)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "open" in df.columns
    assert df["close"].iloc[0] == 101.0


@pytest.mark.asyncio
async def test_rate_limiting(data_fetcher):
    """Test request throttling and rate limit handling."""
    data_fetcher.exchange.fetch_ohlcv = Mock(return_value=[])

    # First request
    await data_fetcher.get_historical_data("BTC/USDT", "1h")

    # Should throttle subsequent requests
    start_time = datetime.now()
    await data_fetcher.get_historical_data("BTC/USDT", "1h")
    elapsed = (datetime.now() - start_time).total_seconds()

    assert (
        elapsed >= 0.09
    )  # Should have throttled (min_interval = 1/10 = 0.1s, allow some tolerance)


@pytest.mark.asyncio
async def test_data_validation(data_fetcher, mock_ohlcv_data):
    """Test OHLCV data validation checks."""
    # Test valid data
    data_fetcher.exchange._exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_data)
    df = await data_fetcher.get_historical_data("BTC/USDT", "1h")
    assert not df.empty

    # Test empty data
    data_fetcher.exchange._exchange.fetch_ohlcv = AsyncMock(return_value=[])
    df = await data_fetcher.get_historical_data("BTC/USDT", "1h")
    assert df.empty

    # Test malformed data (missing close price)
    bad_data = [d[:4] for d in mock_ohlcv_data]  # Remove close price
    data_fetcher.exchange._exchange.fetch_ohlcv = AsyncMock(return_value=bad_data)
    df = await data_fetcher.get_historical_data("BTC/USDT", "1h")
    assert df.empty


@pytest.mark.asyncio
async def test_historical_loader(historical_loader):
    """Test historical data loading and caching."""
    # Setup mock data
    sample_data = pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [98, 99, 100],
            "close": [101, 102, 101],
            "volume": [1000, 1200, 800],
        },
        index=pd.date_range(start="2023-01-01", periods=3, freq="D"),
    )

    historical_loader._load_symbol_data = AsyncMock(return_value=sample_data)

    # Test loading multiple symbols
    data = await historical_loader.load_historical_data(
        symbols=["BTC/USDT", "ETH/USDT"],
        start_date="2023-01-01",
        end_date="2023-01-03",
        timeframe="1d",
    )

    assert isinstance(data, dict)
    assert "BTC/USDT" in data
    assert len(data["BTC/USDT"]) == 3


@pytest.mark.asyncio
async def test_historical_data_validation(historical_loader):
    """Test historical data validation logic."""
    # Valid data
    valid_data = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [102, 103],
            "low": [98, 99],
            "close": [101, 102],
            "volume": [1000, 1200],
        },
        index=pd.date_range(start="2023-01-01", periods=2, freq="D"),
    )

    assert historical_loader._validate_data(valid_data, "1d")

    # Missing column
    invalid_data = valid_data.drop(columns=["close"])
    assert not historical_loader._validate_data(invalid_data, "1d")

    # Invalid timeframe consistency
    invalid_timeframe_data = valid_data.copy()
    invalid_timeframe_data.index = [
        datetime(2023, 1, 1),
        datetime(2023, 1, 3),
    ]  # 2-day gap for 1d timeframe
    assert not historical_loader._validate_data(invalid_timeframe_data, "1d")


@pytest.mark.asyncio
async def test_data_resampling(historical_loader):
    """Test timeframe resampling functionality."""
    hourly_data = pd.DataFrame(
        {
            "open": [100, 101, 102, 101],
            "high": [102, 103, 104, 103],
            "low": [98, 99, 100, 99],
            "close": [101, 102, 101, 100],
            "volume": [1000, 1200, 800, 900],
        },
        index=pd.date_range(start="2023-01-01", periods=4, freq="4h"),
    )

    # Resample 4H -> 1D
    resampled = await historical_loader.resample_data({"BTC/USDT": hourly_data}, "1d")
    daily_data = resampled["BTC/USDT"]

    assert len(daily_data) == 1  # 4 hours x 4 = 1 day
    assert daily_data["high"].iloc[0] == 104  # Should be max of highs
    assert daily_data["volume"].iloc[0] == 3900  # Sum of volumes


@pytest.mark.asyncio
async def test_cache_operations(data_fetcher, mock_ohlcv_data):
    """Test data caching functionality."""
    symbol = "BTC/USDT"
    timeframe = "1h"
    cache_key = data_fetcher._get_cache_key(symbol, timeframe, limit=100)

    # Create test cache directory
    os.makedirs(data_fetcher.cache_dir, exist_ok=True)

    # Test saving to cache
    df = pd.DataFrame(
        mock_ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    await data_fetcher._save_to_cache(cache_key, df)
    cache_path = Path(data_fetcher._cache_dir_path) / f"{cache_key}.json"
    assert cache_path.exists()

    # Test loading from cache
    loaded_df = await data_fetcher._load_from_cache(cache_key)
    assert not loaded_df.empty
    pd.testing.assert_frame_equal(loaded_df, df, check_dtype=True, check_exact=False)

    # Cleanup
    cache_path.unlink()


@pytest.mark.asyncio
async def test_multiple_symbol_fetching(data_fetcher, mock_ohlcv_data):
    """Test concurrent fetching of multiple symbols."""
    data_fetcher.exchange._exchange.fetch_ohlcv = AsyncMock(return_value=mock_ohlcv_data)

    # Test fetching multiple symbols
    data = await data_fetcher.get_multiple_historical_data(
        symbols=["BTC/USDT", "ETH/USDT"], timeframe="1h", limit=3
    )

    assert isinstance(data, dict)
    assert len(data) == 2
    assert all(isinstance(df, pd.DataFrame) for df in data.values())


@pytest.mark.asyncio
async def test_proxy_support(data_fetcher):
    """Test proxy configuration support."""
    proxy_config = {
        "http": "http://proxy.example.com:8080",
        "https": "http://proxy.example.com:8080",
    }
    data_fetcher.config["proxy"] = proxy_config

    # Verify proxy is passed to exchange
    assert data_fetcher.exchange.proxies == proxy_config


@pytest.mark.asyncio
async def test_historical_data_pagination(historical_loader):
    """Test historical data pagination across time windows."""
    mock_fetcher = historical_loader.data_fetcher

    # Setup mock responses for pagination
    date_ranges = [
        (datetime(2023, 1, 1), datetime(2023, 1, 2)),
        (datetime(2023, 1, 2), datetime(2023, 1, 3)),
    ]

    mock_data = [
        pd.DataFrame(
            {"close": [100, 101]}, index=pd.to_datetime(date_ranges[0], utc=True)
        ),
        pd.DataFrame(
            {"close": [102, 103]}, index=pd.to_datetime(date_ranges[1], utc=True)
        ),
    ]

    mock_fetcher.get_historical_data.side_effect = mock_data

    # Test fetching across date range requiring pagination
    data = await historical_loader._fetch_complete_history(
        symbol="BTC/USDT",
        start_date="2023-01-01",
        end_date="2023-01-03",
        timeframe="1d",
    )

    assert len(data) == 3  # Combined data from both pages (with deduplication)
    assert mock_fetcher.get_historical_data.call_count == 2


def test_config_loader_integration():
    """Test that the test config matches DataFetcher expectations."""
    config = {
        "name": "kucoin",
        "api_key": "test_key",
        "api_secret": "test_secret",
        "timeout": 30000,
    }
    fetcher = DataFetcher(config)
    assert fetcher.config["name"] == "kucoin"
