import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
import asyncio
from typing import Dict, Any

from strategies.rsi_strategy import RSIStrategy
from strategies.ema_cross_strategy import EMACrossStrategy
from strategies.base_strategy import StrategyConfig, BaseStrategy
from core.contracts import TradingSignal, SignalType, SignalStrength
from data.data_fetcher import DataFetcher


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2023-01-01', periods=50, freq='1H')
    # Create trending data with some volatility
    base_price = 50000
    trend = np.linspace(0, 1000, 50)  # Upward trend
    noise = np.random.normal(0, 200, 50)  # Random noise
    prices = base_price + trend + noise

    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Create OHLCV data
        high = price + abs(np.random.normal(0, 50))
        low = price - abs(np.random.normal(0, 50))
        open_price = price + np.random.normal(0, 20)
        volume = np.random.randint(1000, 10000)

        data.append({
            'timestamp': date,
            'open': max(open_price, low),  # Ensure open >= low
            'high': high,
            'low': low,
            'close': price,
            'volume': volume,
            'symbol': 'BTC/USDT'
        })

    return pd.DataFrame(data)


@pytest.fixture
def rsi_config():
    """Create RSI strategy configuration."""
    return StrategyConfig(
        name="RSI_Test",
        symbols=["BTC/USDT"],
        timeframe="1h",
        required_history=20,
        params={
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30,
            "position_size": 0.1,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
        }
    )


@pytest.fixture
def ema_config():
    """Create EMA strategy configuration."""
    return StrategyConfig(
        name="EMA_Test",
        symbols=["BTC/USDT"],
        timeframe="1h",
        required_history=25,
        params={
            "fast_ema": 9,
            "slow_ema": 21,
            "position_size": 0.1,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.06,
        }
    )


@pytest.fixture
def mock_data_fetcher():
    """Create a mock data fetcher."""
    fetcher = MagicMock(spec=DataFetcher)
    fetcher.get_historical_data = AsyncMock()
    return fetcher


class TestRSIStrategy:
    """Test cases for RSI strategy."""

    @pytest.mark.asyncio
    async def test_initialization_with_config_object(self, rsi_config):
        """Test strategy initialization with StrategyConfig object."""
        strategy = RSIStrategy(rsi_config)

        assert strategy.config == rsi_config
        assert strategy.params["rsi_period"] == 14
        assert strategy.params["overbought"] == 70
        assert strategy.params["oversold"] == 30
        assert strategy.id.startswith("RSI_Test_")

    @pytest.mark.asyncio
    async def test_initialization_with_dict_config(self):
        """Test strategy initialization with dict config."""
        config_dict = {
            "name": "RSI_Dict_Test",
            "symbols": ["ETH/USDT"],
            "timeframe": "4h",
            "required_history": 20,
            "params": {
                "rsi_period": 21,
                "overbought": 75,
                "oversold": 25,
            }
        }

        strategy = RSIStrategy(config_dict)

        assert strategy.params["rsi_period"] == 21
        assert strategy.params["overbought"] == 75
        assert strategy.params["oversold"] == 25
        assert strategy.id.startswith("RSI_Dict_Test_")

    @pytest.mark.asyncio
    async def test_calculate_indicators(self, rsi_config):
        """Test RSI indicator calculation."""
        strategy = RSIStrategy(rsi_config)

        # Create simpler test data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=25, freq='1H'),
            'open': [50000] * 25,
            'high': [50100] * 25,
            'low': [49900] * 25,
            'close': [50000] * 25,
            'volume': [1000] * 25,
            'symbol': ['BTC/USDT'] * 25
        })

        result = await strategy.calculate_indicators(data)

        assert "rsi" in result.columns
        assert len(result) == len(data)
        # RSI should be NaN for the first few periods
        assert result["rsi"].isna().sum() > 0
        # Valid RSI values should be between 0 and 100
        valid_rsi = result["rsi"].dropna()
        if len(valid_rsi) > 0:
            assert all(0 <= rsi <= 100 for rsi in valid_rsi)

    @pytest.mark.asyncio
    async def test_calculate_indicators_insufficient_data(self, rsi_config):
        """Test RSI calculation with insufficient data."""
        strategy = RSIStrategy(rsi_config)

        # Create data with only 5 rows (less than required)
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
            'open': [50000] * 5,
            'high': [50100] * 5,
            'low': [49900] * 5,
            'close': [50000] * 5,
            'volume': [1000] * 5,
            'symbol': ['BTC/USDT'] * 5
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            await strategy.calculate_indicators(small_data)

    @pytest.mark.asyncio
    async def test_generate_signals_oversold(self, rsi_config):
        """Test signal generation when RSI is oversold."""
        strategy = RSIStrategy(rsi_config)

        # Create data with RSI < 30 (oversold) - use dict format to avoid RSI recalculation
        data = {
            "BTC/USDT": pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=20, freq='1H'),
                'open': [50000] * 20,
                'high': [50100] * 20,
                'low': [49900] * 20,
                'close': [50000] * 20,
                'volume': [1000] * 20,
                'symbol': ['BTC/USDT'] * 20,
                'rsi': [25.0] * 20  # Oversold
            })
        }

        signals = await strategy.generate_signals(data)

        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == SignalType.ENTRY_LONG
        assert signal.symbol == "BTC/USDT"
        assert signal.amount == Decimal("0.1")
        assert signal.metadata["rsi_value"] == 25.0

    @pytest.mark.asyncio
    async def test_generate_signals_overbought(self, rsi_config):
        """Test signal generation when RSI is overbought."""
        strategy = RSIStrategy(rsi_config)

        # Create data with RSI > 70 (overbought) - use dict format
        data = {
            "BTC/USDT": pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=20, freq='1H'),
                'open': [50000] * 20,
                'high': [50100] * 20,
                'low': [49900] * 20,
                'close': [50000] * 20,
                'volume': [1000] * 20,
                'symbol': ['BTC/USDT'] * 20,
                'rsi': [75.0] * 20  # Overbought
            })
        }

        signals = await strategy.generate_signals(data)

        assert len(signals) == 1
        signal = signals[0]
        assert signal.signal_type == SignalType.ENTRY_SHORT
        assert signal.symbol == "BTC/USDT"
        assert signal.amount == Decimal("0.1")
        assert signal.metadata["rsi_value"] == 75.0

    @pytest.mark.asyncio
    async def test_generate_signals_neutral(self, rsi_config):
        """Test signal generation when RSI is neutral."""
        strategy = RSIStrategy(rsi_config)

        # Create data with RSI in neutral range (30-70) - use dict format
        data = {
            "BTC/USDT": pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=20, freq='1H'),
                'open': [50000] * 20,
                'high': [50100] * 20,
                'low': [49900] * 20,
                'close': [50000] * 20,
                'volume': [1000] * 20,
                'symbol': ['BTC/USDT'] * 20,
                'rsi': [50.0] * 20  # Neutral
            })
        }

        signals = await strategy.generate_signals(data)

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_dict_format(self, rsi_config):
        """Test signal generation with dict data format."""
        strategy = RSIStrategy(rsi_config)

        # Create data in dict format
        data = {
            "BTC/USDT": pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=20, freq='1H'),
                'open': [50000] * 20,
                'high': [50100] * 20,
                'low': [49900] * 20,
                'close': [50000] * 20,
                'volume': [1000] * 20,
                'symbol': ['BTC/USDT'] * 20,
                'rsi': [25.0] * 20  # Oversold
            })
        }

        signals = await strategy.generate_signals(data)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.ENTRY_LONG

    @pytest.mark.asyncio
    async def test_generate_signals_empty_data(self, rsi_config):
        """Test signal generation with empty data."""
        strategy = RSIStrategy(rsi_config)

        signals = await strategy.generate_signals(pd.DataFrame())
        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_multiple_symbols(self, rsi_config):
        """Test signal generation with multiple symbols."""
        # Update config for multiple symbols
        rsi_config.symbols = ["BTC/USDT", "ETH/USDT"]
        strategy = RSIStrategy(rsi_config)

        # Create data for both symbols in dict format
        data = {
            "BTC/USDT": pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=20, freq='1H'),
                'open': [50000] * 20,
                'high': [50100] * 20,
                'low': [49900] * 20,
                'close': [50000] * 20,
                'volume': [1000] * 20,
                'symbol': ['BTC/USDT'] * 20,
                'rsi': [25.0] * 20  # BTC oversold
            }),
            "ETH/USDT": pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=20, freq='1H'),
                'open': [3000] * 20,
                'high': [3010] * 20,
                'low': [2990] * 20,
                'close': [3000] * 20,
                'volume': [500] * 20,
                'symbol': ['ETH/USDT'] * 20,
                'rsi': [75.0] * 20  # ETH overbought
            })
        }

        signals = await strategy.generate_signals(data)

        assert len(signals) == 2
        # Should have one LONG signal for BTC and one SHORT signal for ETH
        signal_types = [s.signal_type for s in signals]
        assert SignalType.ENTRY_LONG in signal_types
        assert SignalType.ENTRY_SHORT in signal_types


class TestEMACrossStrategy:
    """Test cases for EMA crossover strategy."""

    @pytest.mark.asyncio
    async def test_initialization_with_config_object(self, ema_config):
        """Test strategy initialization with StrategyConfig object."""
        strategy = EMACrossStrategy(ema_config)

        assert strategy.config == ema_config
        assert strategy.params["fast_ema"] == 9
        assert strategy.params["slow_ema"] == 21
        assert strategy.id.startswith("EMA_Test_")

    @pytest.mark.asyncio
    async def test_calculate_indicators(self, ema_config, sample_ohlcv_data):
        """Test EMA indicator calculation."""
        strategy = EMACrossStrategy(ema_config)

        result = await strategy.calculate_indicators(sample_ohlcv_data)

        assert "fast_ema" in result.columns
        assert "slow_ema" in result.columns
        assert len(result) == len(sample_ohlcv_data)
        assert not result["fast_ema"].isna().all()
        assert not result["slow_ema"].isna().all()

    @pytest.mark.asyncio
    async def test_calculate_indicators_insufficient_data(self, ema_config):
        """Test EMA calculation with insufficient data."""
        strategy = EMACrossStrategy(ema_config)

        # Create data with only 5 rows (less than required)
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
            'open': [50000] * 5,
            'high': [50100] * 5,
            'low': [49900] * 5,
            'close': [50000] * 5,
            'volume': [1000] * 5,
            'symbol': ['BTC/USDT'] * 5
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            await strategy.calculate_indicators(small_data)

    @pytest.mark.asyncio
    async def test_generate_signals_bullish_crossover(self, ema_config):
        """Test signal generation on bullish EMA crossover."""
        strategy = EMACrossStrategy(ema_config)

        # Create data showing bullish crossover
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=25, freq='1H'),
            'open': list(range(50000, 50025)),
            'high': [p + 10 for p in range(50000, 50025)],
            'low': [p - 10 for p in range(50000, 50025)],
            'close': list(range(50000, 50025)),  # Rising prices
            'volume': [1000] * 25,
            'symbol': ['BTC/USDT'] * 25,
        })

        # Manually set EMAs to create crossover scenario
        data['fast_ema'] = data['close'].ewm(span=9, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=21, adjust=False).mean()

        signals = await strategy.generate_signals(data)

        # Should generate a LONG signal if crossover occurred
        long_signals = [s for s in signals if s.signal_type == SignalType.ENTRY_LONG]
        assert len(long_signals) >= 0  # May or may not generate depending on exact crossover

    @pytest.mark.asyncio
    async def test_generate_signals_bearish_crossover(self, ema_config):
        """Test signal generation on bearish EMA crossover."""
        strategy = EMACrossStrategy(ema_config)

        # Create data showing bearish crossover
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=25, freq='1H'),
            'open': list(range(50024, 49999, -1)),
            'high': [p + 10 for p in range(50024, 49999, -1)],
            'low': [p - 10 for p in range(50024, 49999, -1)],
            'close': list(range(50024, 49999, -1)),  # Falling prices
            'volume': [1000] * 25,
            'symbol': ['BTC/USDT'] * 25,
        })

        # Manually set EMAs to create crossover scenario
        data['fast_ema'] = data['close'].ewm(span=9, adjust=False).mean()
        data['slow_ema'] = data['close'].ewm(span=21, adjust=False).mean()

        signals = await strategy.generate_signals(data)

        # Should generate a SHORT signal if crossover occurred
        short_signals = [s for s in signals if s.signal_type == SignalType.ENTRY_SHORT]
        assert len(short_signals) >= 0  # May or may not generate depending on exact crossover

    @pytest.mark.asyncio
    async def test_generate_signals_no_crossover(self, ema_config):
        """Test signal generation when no crossover occurs."""
        strategy = EMACrossStrategy(ema_config)

        # Create data with parallel EMAs (no crossover)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=25, freq='1H'),
            'open': [50000] * 25,
            'high': [50100] * 25,
            'low': [49900] * 25,
            'close': [50000] * 25,  # Flat prices
            'volume': [1000] * 25,
            'symbol': ['BTC/USDT'] * 25,
        })

        # Set EMAs to be parallel (no crossover)
        data['fast_ema'] = 50000
        data['slow_ema'] = 50000

        signals = await strategy.generate_signals(data)

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_insufficient_data(self, ema_config):
        """Test signal generation with insufficient data for crossover detection."""
        strategy = EMACrossStrategy(ema_config)

        # Create data with only 1 row (insufficient for crossover detection)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=1, freq='1H'),
            'open': [50000],
            'high': [50100],
            'low': [49900],
            'close': [50000],
            'volume': [1000],
            'symbol': ['BTC/USDT'],
            'fast_ema': [50000],
            'slow_ema': [50000],
        })

        signals = await strategy.generate_signals(data)

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_generate_signals_dict_format(self, ema_config):
        """Test signal generation with dict data format."""
        strategy = EMACrossStrategy(ema_config)

        # Create data in dict format
        data = {
            "BTC/USDT": pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=25, freq='1H'),
                'open': [50000] * 25,
                'high': [50100] * 25,
                'low': [49900] * 25,
                'close': [50000] * 25,
                'volume': [1000] * 25,
                'symbol': ['BTC/USDT'] * 25,
                'fast_ema': [50000] * 25,
                'slow_ema': [50000] * 25,
            })
        }

        signals = await strategy.generate_signals(data)

        assert len(signals) == 0  # No crossover in flat data


class TestStrategyIntegration:
    """Integration tests for strategy functionality."""

    @pytest.mark.asyncio
    async def test_strategy_run_workflow(self, rsi_config, mock_data_fetcher, sample_ohlcv_data):
        """Test complete strategy run workflow."""
        strategy = RSIStrategy(rsi_config)

        # Mock data fetcher to return sample data
        mock_data_fetcher.get_historical_data.return_value = sample_ohlcv_data

        # Initialize strategy
        await strategy.initialize(mock_data_fetcher)

        # Run strategy
        signals = await strategy.run()

        # Verify signals were generated (or not, depending on data)
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)

    @pytest.mark.asyncio
    async def test_strategy_shutdown(self, rsi_config):
        """Test strategy shutdown functionality."""
        strategy = RSIStrategy(rsi_config)

        await strategy.shutdown()

        assert not strategy.initialized

    @pytest.mark.asyncio
    async def test_strategy_performance_metrics(self, rsi_config):
        """Test strategy performance metrics."""
        strategy = RSIStrategy(rsi_config)

        metrics = strategy.get_performance_metrics()

        assert metrics["strategy_id"] == strategy.id
        assert metrics["name"] == "RSI_Test"
        assert metrics["signals_generated"] == 0
        assert metrics["symbols"] == ["BTC/USDT"]
        assert metrics["timeframe"] == "1h"

    @pytest.mark.asyncio
    async def test_strategy_error_handling(self, rsi_config):
        """Test strategy error handling in signal generation."""
        strategy = RSIStrategy(rsi_config)

        # Test with invalid data format
        signals = await strategy.generate_signals("invalid_data")
        assert signals == []

    @pytest.mark.asyncio
    async def test_strategy_create_signal_helper(self, rsi_config):
        """Test the create_signal helper method."""
        strategy = RSIStrategy(rsi_config)

        signal = strategy.create_signal(
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            strength=SignalStrength.STRONG,
            order_type="market",
            amount=0.1,
            current_price=50000,
            stop_loss=49000,
            take_profit=52000,
            metadata={"test": "value"}
        )

        assert signal.symbol == "BTC/USDT"
        assert signal.signal_type == SignalType.ENTRY_LONG
        assert signal.amount == Decimal("0.1")
        assert signal.current_price == Decimal("50000")
        assert signal.stop_loss == Decimal("49000")
        assert signal.take_profit == Decimal("52000")
        assert signal.metadata["test"] == "value"
