"""
tests/test_strategy.py

Unit tests for strategy functionality including signal generation,
indicator calculation, and strategy lifecycle management.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock
from typing import List
import numpy as np
from datetime import datetime

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import OrderType
from data.data_fetcher import DataFetcher
from decimal import Decimal
from strategies.mixins import TrendAnalysisMixin, VolatilityAnalysisMixin


@pytest.fixture
def sample_ohlcv_data():
    """Fixture providing sample OHLCV data for testing."""
    return pd.DataFrame(
        {
            "open": [100, 101, 102, 101, 100, 99, 98, 99, 100, 101],
            "high": [102, 103, 104, 103, 102, 101, 100, 101, 102, 103],
            "low": [98, 99, 100, 99, 98, 97, 96, 97, 98, 99],
            "close": [101, 102, 101, 100, 99, 98, 99, 100, 101, 102],
            "volume": [1000, 1200, 800, 900, 1100, 1300, 1400, 1200, 1100, 1000],
        },
        index=pd.date_range(start="2023-01-01", periods=10, freq="D"),
    )


@pytest.fixture
def strategy_config():
    """Fixture providing base strategy configuration."""
    return StrategyConfig(
        name="TestStrategy",
        symbols=["BTC/USDT"],
        timeframe="1d",
        required_history=10,
        params={"param1": 10, "param2": 20},
    )


@pytest.fixture
def mock_data_fetcher(sample_ohlcv_data):
    """Fixture providing mocked DataFetcher instance."""
    mock = AsyncMock(spec=DataFetcher)
    mock.get_historical_data.return_value = sample_ohlcv_data
    return mock


class ConcreteStrategy(BaseStrategy):
    """Concrete strategy implementation for testing base class functionality."""

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Test indicator calculation that adds simple moving averages."""
        data["sma_5"] = data["close"].rolling(5).mean()
        data["sma_10"] = data["close"].rolling(10).mean()
        return data

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Test signal generation that produces mock signals."""
        if len(data) < self.config.required_history:
            return []

        last_row = data.iloc[-1]
        return [
            TradingSignal(
                strategy_id=self.id,
                symbol=self.config.symbols[0],
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.1,  # Will be calculated by risk manager
                price=float(last_row["close"]),
                current_price=float(last_row["close"]),
                stop_loss=float(last_row["close"] * 0.95),
                metadata={"test": True},
            )
        ]


@pytest.fixture
def concrete_strategy(strategy_config, mock_data_fetcher):
    """Fixture providing initialized concrete strategy instance."""
    strategy = ConcreteStrategy(strategy_config)
    return strategy


@pytest.mark.asyncio
async def test_strategy_initialization(concrete_strategy, strategy_config):
    """Test strategy initialization with config."""
    assert concrete_strategy.config.name == strategy_config.name
    assert concrete_strategy.config.symbols == strategy_config.symbols
    assert concrete_strategy.config.timeframe == strategy_config.timeframe
    assert concrete_strategy.initialized is False


@pytest.mark.asyncio
async def test_strategy_lifecycle(concrete_strategy, mock_data_fetcher):
    """Test full strategy lifecycle including initialization and shutdown."""
    # Initialize strategy
    await concrete_strategy.initialize(mock_data_fetcher)
    assert concrete_strategy.initialized is True
    assert concrete_strategy.data_fetcher == mock_data_fetcher

    # Run strategy
    signals = await concrete_strategy.run()
    assert len(signals) == 1
    assert signals[0].symbol == "BTC/USDT"

    # Shutdown strategy
    await concrete_strategy.shutdown()
    assert concrete_strategy.initialized is False


@pytest.mark.asyncio
async def test_indicator_calculation(concrete_strategy, sample_ohlcv_data):
    """Test indicator calculation logic."""
    # Test with sufficient data
    data_with_indicators = await concrete_strategy.calculate_indicators(
        sample_ohlcv_data
    )
    assert "sma_5" in data_with_indicators.columns
    assert "sma_10" in data_with_indicators.columns
    assert (
        not data_with_indicators["sma_5"].iloc[:4].notna().any()
    )  # First 4 should be NaN
    assert (
        data_with_indicators["sma_5"].iloc[4:].notna().all()
    )  # After 5 should have values

    # Test with insufficient data
    small_data = sample_ohlcv_data.iloc[:3]
    with pytest.raises(ValueError):
        await concrete_strategy.calculate_indicators(small_data)


@pytest.mark.asyncio
async def test_signal_generation(concrete_strategy, sample_ohlcv_data):
    """Test signal generation logic."""
    # Test with sufficient data
    data_with_indicators = await concrete_strategy.calculate_indicators(
        sample_ohlcv_data
    )
    signals = await concrete_strategy.generate_signals(data_with_indicators)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.symbol == "BTC/USDT"
    assert signal.signal_type == SignalType.ENTRY_LONG
    assert signal.order_type == OrderType.MARKET
    assert signal.metadata["test"] is True

    # Test with insufficient data
    small_data = sample_ohlcv_data.iloc[:5]
    signals = await concrete_strategy.generate_signals(small_data)
    assert len(signals) == 0


@pytest.mark.asyncio
async def test_strategy_run_method(concrete_strategy, mock_data_fetcher):
    """Test the main run method that orchestrates the strategy."""
    await concrete_strategy.initialize(mock_data_fetcher)

    # Successful run
    signals = await concrete_strategy.run()
    assert len(signals) == 1

    # Failed data fetch (returns empty DataFrame)
    mock_data_fetcher.get_historical_data.return_value = pd.DataFrame()
    signals = await concrete_strategy.run()
    assert len(signals) == 0


@pytest.mark.asyncio
async def test_strategy_performance_tracking(concrete_strategy, mock_data_fetcher):
    """Test strategy performance tracking metrics."""
    await concrete_strategy.initialize(mock_data_fetcher)

    # Initial state
    metrics = concrete_strategy.get_performance_metrics()
    assert metrics["signals_generated"] == 0

    # Generate some signals
    for _ in range(3):
        await concrete_strategy.run()

    # Verify metrics updated
    metrics = concrete_strategy.get_performance_metrics()
    assert metrics["signals_generated"] == 3
    assert metrics["last_signal_time"] > 0


@pytest.mark.asyncio
async def test_trend_analysis_mixin():
    """Test trend analysis mixin methods."""

    # Create strategy with mixin
    class TrendStrategy(BaseStrategy, TrendAnalysisMixin):
        def __init__(self, config):
            super().__init__(config)

        async def calculate_indicators(self, data):
            return data

        async def generate_signals(self, data):
            return []

    config = StrategyConfig(
        name="TrendStrat", symbols=["BTC/USDT"], timeframe="1d", required_history=20
    )
    strategy = TrendStrategy(config)

    # Test trend strength calculation
    prices = pd.Series([100, 101, 102, 103, 104, 103, 102, 103, 104, 105])
    strength = await strategy.calculate_trend_strength(prices, period=5)
    assert 0 <= strength <= 1

    # Test support/resistance identification
    ohlc_data = pd.DataFrame(
        {
            "high": [102, 103, 104, 103, 102],
            "low": [98, 99, 100, 99, 98],
            "close": [101, 102, 101, 100, 99],
        }
    )
    levels = await strategy.identify_support_resistance(ohlc_data)
    assert "support" in levels
    assert "resistance" in levels
    assert 0 <= levels["current_position"] <= 1


@pytest.mark.asyncio
async def test_volatility_analysis_mixin():
    """Test volatility analysis mixin methods."""

    # Create strategy with mixin
    class VolatilityStrategy(BaseStrategy, VolatilityAnalysisMixin):
        def __init__(self, config):
            super().__init__(config)

        async def calculate_indicators(self, data):
            return data

        async def generate_signals(self, data):
            return []

    config = StrategyConfig(
        name="VolatilityStrat",
        symbols=["BTC/USDT"],
        timeframe="1d",
        required_history=20,
    )
    strategy = VolatilityStrategy(config)

    # Test ATR calculation
    ohlc_data = pd.DataFrame(
        {
            "high": [102, 103, 104, 103, 102],
            "low": [98, 99, 100, 99, 98],
            "close": [101, 102, 101, 100, 99],
        }
    )
    atr = await strategy.calculate_atr(ohlc_data, period=3)
    assert atr > 0

    # Test volatility calculation
    prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101])
    volatility = await strategy.calculate_volatility(prices, period=5)
    assert volatility > 0


def test_strategy_config_validation():
    """Test strategy configuration validation."""
    # Valid config
    valid_config = StrategyConfig(
        name="ValidStrategy", symbols=["BTC/USDT"], timeframe="1h", required_history=50
    )
    assert valid_config.name == "ValidStrategy"

    # Invalid config (missing required fields)
    with pytest.raises(ValueError):
        StrategyConfig(
            name="",  # Empty name
            symbols=["BTC/USDT"],
            timeframe="1h",
            required_history=50,
        )


@pytest.mark.asyncio
async def test_strategy_error_handling(concrete_strategy, mock_data_fetcher):
    """Test strategy error handling during execution."""
    await concrete_strategy.initialize(mock_data_fetcher)

    # Force an error in indicator calculation
    with patch.object(
        concrete_strategy, "calculate_indicators", side_effect=Exception("Test error")
    ):
        signals = await concrete_strategy.run()
        assert len(signals) == 0  # Should return empty list on error

    # Force an error in signal generation
    with patch.object(
        concrete_strategy, "generate_signals", side_effect=Exception("Test error")
    ):
        signals = await concrete_strategy.run()
        assert len(signals) == 0  # Should return empty list on error


@pytest.mark.asyncio
async def test_strategy_create_signal(concrete_strategy):
    """Test the create_signal helper method."""
    signal = concrete_strategy.create_signal(
        symbol="ETH/USDT",
        signal_type=SignalType.ENTRY_SHORT,
        strength=SignalStrength.MODERATE,
        order_type=OrderType.LIMIT,
        amount=0.5,
        price=1500.0,
        current_price=1520.0,
        stop_loss=1550.0,
        take_profit=1450.0,
        metadata={"test": True},
    )

    assert signal.strategy_id == concrete_strategy.id
    assert signal.symbol == "ETH/USDT"
    assert signal.signal_type == SignalType.ENTRY_SHORT
    assert signal.order_type == OrderType.LIMIT
    assert signal.amount == Decimal("0.5")
    assert signal.price == Decimal("1500.0")
    assert signal.stop_loss == Decimal("1550.0")
    assert signal.metadata["test"] is True
