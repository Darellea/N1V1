"""
Comprehensive Integration Tests for N1V1 Crypto Trading Framework Strategy Components

This module contains comprehensive integration tests that verify the end-to-end
functionality of trading strategies, including data fetching, indicator calculation,
signal generation, and risk management interactions.

Tests cover:
- End-to-end strategy execution workflow
- Component interactions between strategies, data fetchers, and risk managers
- Output validation and correctness verification
- Error scenarios and graceful failure handling
- Multi-symbol and multi-timeframe scenarios
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from strategies.base_strategy import BaseStrategy, StrategyConfig
from strategies.rsi_strategy import RSIStrategy
from strategies.bollinger_reversion_strategy import BollingerReversionStrategy
from strategies.atr_breakout_strategy import ATRBreakoutStrategy
from core.contracts import TradingSignal, SignalType, SignalStrength
from risk.risk_manager import RiskManager
from data.data_fetcher import DataFetcher


class MockDataFetcher:
    """Mock data fetcher for integration testing."""

    def __init__(self, test_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize with test data or generate mock data."""
        self.test_data = test_data or self._generate_mock_data()
        self.call_count = 0
        self.last_symbol = None
        self.last_timeframe = None

    def _generate_mock_data(self) -> Dict[str, pd.DataFrame]:
        """Generate realistic mock OHLCV data for testing."""
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')

        # Generate BTC data with trend and volatility
        np.random.seed(42)
        base_price = 50000
        returns = np.random.normal(0.0001, 0.02, len(dates))
        btc_prices = base_price * np.exp(np.cumsum(returns))

        btc_data = pd.DataFrame({
            'open': btc_prices,
            'high': btc_prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': btc_prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': btc_prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'volume': np.random.uniform(100, 1000, len(dates))
        }, index=dates)

        # Ensure high >= max(open, close) and low <= min(open, close)
        btc_data['high'] = np.maximum(btc_data[['open', 'close']].max(axis=1), btc_data['high'])
        btc_data['low'] = np.minimum(btc_data[['open', 'close']].min(axis=1), btc_data['low'])

        # Generate ETH data
        eth_prices = btc_prices * 0.1  # Different scale
        eth_data = pd.DataFrame({
            'open': eth_prices,
            'high': eth_prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': eth_prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': eth_prices * (1 + np.random.normal(0, 0.01, len(dates))),
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)

        eth_data['high'] = np.maximum(eth_data[['open', 'close']].max(axis=1), eth_data['high'])
        eth_data['low'] = np.minimum(eth_data[['open', 'close']].min(axis=1), eth_data['low'])

        return {
            'BTC/USDT': btc_data,
            'ETH/USDT': eth_data
        }

    async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Mock historical data retrieval."""
        self.call_count += 1
        self.last_symbol = symbol
        self.last_timeframe = timeframe

        if symbol not in self.test_data:
            return None

        data = self.test_data[symbol]
        if len(data) > limit:
            return data.tail(limit)
        return data

    async def get_multiple_historical_data(self, symbols: List[str], timeframe: str = '1h', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Mock multiple symbol data retrieval."""
        result = {}
        for symbol in symbols:
            data = await self.get_historical_data(symbol, timeframe, limit)
            if data is not None:
                result[symbol] = data
        return result


@pytest.fixture
def mock_data_fetcher():
    """Fixture providing mock data fetcher."""
    return MockDataFetcher()


@pytest.fixture
def rsi_strategy_config():
    """Fixture providing RSI strategy configuration."""
    return StrategyConfig(
        name="RSI Strategy",
        symbols=["BTC/USDT"],
        timeframe="1h",
        required_history=50,
        params={
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30,
            "position_size": 0.1,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.1,
            "volume_period": 10,
            "volume_threshold": 1.5
        }
    )


@pytest.fixture
def risk_manager_config():
    """Fixture providing risk manager configuration."""
    return {
        "max_position_size": Decimal("0.3"),
        "max_daily_drawdown": Decimal("0.1"),
        "risk_reward_ratio": Decimal("2.0"),
        "position_sizing_method": "fixed_percent",
        "fixed_percent": Decimal("0.1"),
        "require_stop_loss": True,
        "stop_loss_method": "percentage",
        "stop_loss_percentage": Decimal("0.02")
    }


class TestStrategyIntegrationWorkflow:
    """Integration tests for complete strategy execution workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_rsi_strategy_workflow(self, mock_data_fetcher, rsi_strategy_config):
        """
        Test complete end-to-end workflow for RSI strategy.

        This test verifies that a strategy can:
        1. Initialize with data fetcher
        2. Fetch market data
        3. Calculate indicators
        4. Generate trading signals
        5. Return properly formatted signals
        """
        # Initialize strategy
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(mock_data_fetcher)

        # Verify initialization
        assert strategy.initialized is True
        assert strategy.data_fetcher == mock_data_fetcher

        # Run complete strategy workflow
        signals = await strategy.run()

        # Verify results
        assert isinstance(signals, list)

        # Check signal format if any signals were generated
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "BTC/USDT"
            assert signal.strategy_id.startswith("RSI Strategy_")
            assert signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]
            assert signal.signal_strength == SignalStrength.STRONG
            assert signal.order_type == "market"
            assert signal.amount > 0
            assert signal.current_price > 0
            assert signal.metadata is not None
            assert "rsi_value" in signal.metadata

        # Verify data fetcher was called
        assert mock_data_fetcher.call_count > 0
        assert mock_data_fetcher.last_symbol == "BTC/USDT"
        assert mock_data_fetcher.last_timeframe == "1h"

    @pytest.mark.asyncio
    async def test_strategy_with_multi_symbol_workflow(self, mock_data_fetcher):
        """
        Test strategy workflow with multiple symbols.

        Verifies that strategies can handle multiple symbols correctly
        and generate appropriate signals for each.
        """
        # Configure strategy for multiple symbols
        config = StrategyConfig(
            name="Multi-Symbol RSI Strategy",
            symbols=["BTC/USDT", "ETH/USDT"],
            timeframe="1h",
            required_history=50,
            params={
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30,
                "position_size": 0.1,
                "volume_threshold": 1.0  # Lower threshold for testing
            }
        )

        strategy = RSIStrategy(config)
        await strategy.initialize(mock_data_fetcher)

        # Run strategy
        signals = await strategy.run()

        # Verify results
        assert isinstance(signals, list)

        # Check that signals are generated for appropriate symbols
        signal_symbols = {signal.symbol for signal in signals}
        assert signal_symbols.issubset({"BTC/USDT", "ETH/USDT"})

        # Verify data fetcher calls
        assert mock_data_fetcher.call_count >= 2  # At least one call per symbol

    @pytest.mark.asyncio
    async def test_strategy_error_handling_workflow(self, rsi_strategy_config):
        """
        Test strategy error handling in integration workflow.

        Verifies that strategies handle errors gracefully and continue
        processing when possible.
        """
        # Create mock data fetcher that sometimes fails
        class FailingDataFetcher(MockDataFetcher):
            def __init__(self):
                super().__init__()
                self.fail_count = 0

            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                self.fail_count += 1
                if self.fail_count == 1:
                    # Fail on first call
                    raise ConnectionError("Network error")
                return await super().get_historical_data(symbol, timeframe, limit)

        failing_fetcher = FailingDataFetcher()
        strategy = RSIStrategy(rsi_strategy_config)

        # Initialize with failing fetcher
        await strategy.initialize(failing_fetcher)

        # Run strategy - should handle error gracefully
        signals = await strategy.run()

        # Should still return a list (possibly empty) despite error
        assert isinstance(signals, list)

        # Verify fetcher was called multiple times (retry behavior)
        assert failing_fetcher.call_count >= 2

    @pytest.mark.asyncio
    async def test_strategy_insufficient_data_handling(self, rsi_strategy_config):
        """
        Test strategy behavior with insufficient data.

        Verifies that strategies handle cases where there's not enough
        historical data for indicator calculation.
        """
        # Create data fetcher with minimal data
        minimal_data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0],
            'high': [101.0, 102.0, 103.0],
            'low': [99.0, 100.0, 101.0],
            'open': [100.0, 101.0, 102.0],
            'volume': [1000, 1000, 1000]
        })

        class MinimalDataFetcher:
            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                return minimal_data

        fetcher = MinimalDataFetcher()
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(fetcher)

        # Run strategy with insufficient data
        signals = await strategy.run()

        # Should handle gracefully (likely return empty list due to insufficient data)
        assert isinstance(signals, list)


class TestStrategyRiskManagerIntegration:
    """Integration tests for strategy and risk manager interactions."""

    @pytest.mark.asyncio
    async def test_strategy_signal_risk_validation(self, mock_data_fetcher, rsi_strategy_config, risk_manager_config):
        """
        Test integration between strategy signal generation and risk validation.

        Verifies that signals generated by strategies pass through risk
        management validation correctly.
        """
        # Initialize strategy
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(mock_data_fetcher)

        # Initialize risk manager
        risk_manager = RiskManager(risk_manager_config)

        # Generate signals from strategy
        signals = await strategy.run()

        # Test each signal against risk manager
        valid_signals = []
        for signal in signals:
            # Get market data for risk evaluation
            market_data = await mock_data_fetcher.get_historical_data(
                signal.symbol, rsi_strategy_config.timeframe, 50
            )

            # Convert DataFrame to dict format expected by risk manager
            if market_data is not None:
                market_dict = {
                    'close': market_data['close'].values,
                    'high': market_data['high'].values,
                    'low': market_data['low'].values,
                    'open': market_data['open'].values,
                    'volume': market_data['volume'].values
                }
            else:
                market_dict = None

            # Evaluate signal with risk manager
            is_valid = await risk_manager.evaluate_signal(signal, market_dict)

            if is_valid:
                valid_signals.append(signal)

        # Verify that risk validation works
        assert len(valid_signals) <= len(signals)  # Risk manager may reject some signals

        # Verify valid signals have required risk management fields
        for signal in valid_signals:
            assert signal.stop_loss is not None
            assert signal.take_profit is not None
            assert signal.amount > 0

    @pytest.mark.asyncio
    async def test_risk_manager_position_sizing_integration(self, mock_data_fetcher, rsi_strategy_config, risk_manager_config):
        """
        Test risk manager position sizing with strategy signals.

        Verifies that position sizing calculations work correctly
        with real market data from strategies.
        """
        # Initialize components
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(mock_data_fetcher)
        risk_manager = RiskManager(risk_manager_config)

        # Get market data
        market_data = await mock_data_fetcher.get_historical_data("BTC/USDT", "1h", 50)

        # Create a test signal
        test_signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("0"),  # Zero to trigger position sizing calculation
            current_price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc)
        )

        # Convert market data for risk manager
        market_dict = None
        if market_data is not None:
            market_dict = {
                'close': market_data['close'].values[-20:],  # Last 20 periods
                'high': market_data['high'].values[-20:],
                'low': market_data['low'].values[-20:],
                'open': market_data['open'].values[-20:],
                'volume': market_data['volume'].values[-20:]
            }

        # Calculate position size
        position_size = await risk_manager.calculate_position_size(test_signal, market_dict)

        # Verify position size is reasonable
        assert position_size > 0
        assert position_size <= Decimal("3000")  # Max 30% of $10k account

    @pytest.mark.asyncio
    async def test_risk_manager_stop_loss_integration(self, mock_data_fetcher, risk_manager_config):
        """
        Test risk manager stop loss calculation with market data.

        Verifies that stop loss calculations work correctly with
        real market data patterns.
        """
        risk_manager = RiskManager(risk_manager_config)

        # Get market data
        market_data = await mock_data_fetcher.get_historical_data("BTC/USDT", "1h", 50)

        # Create test signal
        test_signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1000"),
            current_price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc)
        )

        # Convert market data
        market_dict = None
        if market_data is not None:
            market_dict = {
                'close': market_data['close'].values[-20:],
                'high': market_data['high'].values[-20:],
                'low': market_data['low'].values[-20:],
                'open': market_data['open'].values[-20:],
                'volume': market_data['volume'].values[-20:]
            }

        # Calculate stop loss
        stop_loss = await risk_manager.calculate_dynamic_stop_loss(test_signal, market_dict)

        # Verify stop loss is calculated and reasonable
        assert stop_loss is not None
        assert stop_loss < test_signal.current_price  # For long position

        # Stop loss should be within reasonable range
        loss_pct = (test_signal.current_price - stop_loss) / test_signal.current_price
        assert Decimal("0.01") <= loss_pct <= Decimal("0.1")  # 1% to 10% stop loss

    @pytest.mark.asyncio
    async def test_portfolio_risk_limits_integration(self, mock_data_fetcher, rsi_strategy_config, risk_manager_config):
        """
        Test portfolio-level risk limits with multiple strategy signals.

        Verifies that risk manager enforces portfolio-level constraints
        across multiple signals.
        """
        # Initialize components
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(mock_data_fetcher)
        risk_manager = RiskManager(risk_manager_config)

        # Generate multiple signals
        signals = await strategy.run()

        # Get market data for validation
        market_data = await mock_data_fetcher.get_historical_data("BTC/USDT", "1h", 50)
        market_dict = None
        if market_data is not None:
            market_dict = {
                'close': market_data['close'].values[-20:],
                'high': market_data['high'].values[-20:],
                'low': market_data['low'].values[-20:],
                'open': market_data['open'].values[-20:],
                'volume': market_data['volume'].values[-20:]
            }

        # Evaluate signals and track total exposure
        total_exposure = Decimal("0")
        valid_signals = []

        for signal in signals:
            is_valid = await risk_manager.evaluate_signal(signal, market_dict)
            if is_valid:
                valid_signals.append(signal)
                total_exposure += signal.amount

        # Verify portfolio limits are respected
        max_allowed = risk_manager.max_position_size * Decimal("10000")  # 30% of $10k
        assert total_exposure <= max_allowed

        # Verify no single position exceeds limits
        for signal in valid_signals:
            assert signal.amount <= max_allowed


class TestStrategyDataFetcherIntegration:
    """Integration tests for strategy and data fetcher interactions."""

    @pytest.mark.asyncio
    async def test_data_fetcher_caching_integration(self, rsi_strategy_config):
        """
        Test data fetcher caching behavior with strategy usage.

        Verifies that data fetcher caching works correctly and
        reduces redundant data requests.
        """
        # Create data fetcher with caching simulation
        class CachingDataFetcher(MockDataFetcher):
            def __init__(self):
                super().__init__()
                self.cache = {}
                self.cache_hits = 0
                self.cache_misses = 0

            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                cache_key = f"{symbol}_{timeframe}_{limit}"

                if cache_key in self.cache:
                    self.cache_hits += 1
                    return self.cache[cache_key]
                else:
                    self.cache_misses += 1
                    data = await super().get_historical_data(symbol, timeframe, limit)
                    self.cache[cache_key] = data
                    return data

        caching_fetcher = CachingDataFetcher()

        # Initialize strategy
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(caching_fetcher)

        # Run strategy multiple times
        for _ in range(3):
            signals = await strategy.run()

        # Verify caching behavior
        assert caching_fetcher.cache_hits > 0
        assert caching_fetcher.cache_misses > 0
        assert caching_fetcher.cache_hits >= caching_fetcher.cache_misses

    @pytest.mark.asyncio
    async def test_data_fetcher_error_recovery_integration(self, rsi_strategy_config):
        """
        Test data fetcher error recovery with strategy execution.

        Verifies that strategies can recover from data fetcher errors
        and continue processing.
        """
        # Create unreliable data fetcher
        class UnreliableDataFetcher(MockDataFetcher):
            def __init__(self):
                super().__init__()
                self.error_count = 0
                self.max_errors = 2

            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                if self.error_count < self.max_errors:
                    self.error_count += 1
                    raise ConnectionError(f"Simulated network error {self.error_count}")

                return await super().get_historical_data(symbol, timeframe, limit)

        unreliable_fetcher = UnreliableDataFetcher()
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(unreliable_fetcher)

        # Run strategy - should eventually succeed after retries
        signals = await strategy.run()

        # Verify strategy handled errors gracefully
        assert isinstance(signals, list)
        assert unreliable_fetcher.error_count == unreliable_fetcher.max_errors

    @pytest.mark.asyncio
    async def test_multi_timeframe_data_integration(self, rsi_strategy_config):
        """
        Test multi-timeframe data handling in strategy integration.

        Verifies that strategies can work with multi-timeframe data
        when available from the data fetcher.
        """
        # Create data fetcher with multi-timeframe support
        class MultiTimeframeDataFetcher(MockDataFetcher):
            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                # Simulate different data based on timeframe
                if timeframe == "4h":
                    # Aggregate hourly data to 4-hour
                    hourly_data = await super().get_historical_data(symbol, "1h", limit * 4)
                    if hourly_data is not None:
                        # Simple aggregation (in real implementation would be more sophisticated)
                        return hourly_data.resample('4H').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()

                return await super().get_historical_data(symbol, timeframe, limit)

        mt_fetcher = MultiTimeframeDataFetcher()
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(mt_fetcher)

        # Run strategy
        signals = await strategy.run()

        # Verify strategy works with multi-timeframe data
        assert isinstance(signals, list)


class TestErrorScenariosIntegration:
    """Integration tests for error scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_corrupted_market_data_integration(self, rsi_strategy_config):
        """
        Test strategy behavior with corrupted market data.

        Verifies that strategies handle corrupted or invalid market data
        gracefully without crashing.
        """
        # Create data fetcher with corrupted data
        class CorruptedDataFetcher:
            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                # Return data with NaN values and invalid data
                dates = pd.date_range('2023-01-01', periods=50, freq='1h')
                corrupted_data = pd.DataFrame({
                    'close': [100.0 if i % 5 != 0 else np.nan for i in range(50)],  # Some NaN values
                    'high': [101.0 if i % 7 != 0 else np.nan for i in range(50)],
                    'low': [99.0 if i % 3 != 0 else np.nan for i in range(50)],
                    'open': [100.0] * 50,
                    'volume': [-100 if i % 10 == 0 else 1000 for i in range(50)]  # Some negative volumes
                }, index=dates)

                return corrupted_data

        corrupted_fetcher = CorruptedDataFetcher()
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(corrupted_fetcher)

        # Run strategy with corrupted data
        signals = await strategy.run()

        # Should handle corrupted data gracefully
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_empty_market_data_integration(self, rsi_strategy_config):
        """
        Test strategy behavior with empty market data.

        Verifies that strategies handle empty data responses
        from data fetchers appropriately.
        """
        # Create data fetcher that returns empty data
        class EmptyDataFetcher:
            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                return pd.DataFrame()  # Empty DataFrame

        empty_fetcher = EmptyDataFetcher()
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(empty_fetcher)

        # Run strategy with empty data
        signals = await strategy.run()

        # Should return empty signal list
        assert signals == []

    @pytest.mark.asyncio
    async def test_network_timeout_integration(self, rsi_strategy_config):
        """
        Test strategy behavior with network timeouts.

        Verifies that strategies handle network timeouts and
        connection errors appropriately.
        """
        import asyncio

        # Create data fetcher that simulates timeouts
        class TimeoutDataFetcher:
            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                await asyncio.sleep(0.1)  # Simulate delay
                raise asyncio.TimeoutError("Network timeout")

        timeout_fetcher = TimeoutDataFetcher()
        strategy = RSIStrategy(rsi_strategy_config)
        await strategy.initialize(timeout_fetcher)

        # Run strategy with timeout
        signals = await strategy.run()

        # Should handle timeout gracefully
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_risk_manager_circuit_breaker_integration(self, mock_data_fetcher, risk_manager_config):
        """
        Test risk manager circuit breaker behavior in integration.

        Verifies that circuit breakers protect against cascading failures
        in risk management calculations.
        """
        risk_manager = RiskManager(risk_manager_config)

        # Create signal that will cause repeated failures
        test_signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1000"),
            current_price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc)
        )

        # Get market data
        market_data = await mock_data_fetcher.get_historical_data("BTC/USDT", "1h", 50)
        market_dict = None
        if market_data is not None:
            market_dict = {
                'close': market_data['close'].values[-20:],
                'high': market_data['high'].values[-20:],
                'low': market_data['low'].values[-20:],
                'open': market_data['open'].values[-20:],
                'volume': market_data['volume'].values[-20:]
            }

        # Test circuit breaker by simulating failures
        with patch.object(risk_manager, '_get_current_balance', side_effect=Exception("Simulated failure")):
            # First few calls should fail but not trigger circuit breaker
            for _ in range(3):
                result = await risk_manager.evaluate_signal(test_signal, market_dict)
                assert result is False  # Should fail due to balance error

        # Verify circuit breaker stats
        stats = risk_manager.get_circuit_breaker_stats()
        assert 'position_size_circuit_breaker' in stats


class TestPerformanceIntegration:
    """Integration tests for performance and scalability."""

    @pytest.mark.asyncio
    async def test_strategy_performance_with_large_dataset(self):
        """
        Test strategy performance with large datasets.

        Verifies that strategies can handle large amounts of data
        efficiently without excessive memory usage or timeouts.
        """
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=5000, freq='1h')  # ~6 months of data
        large_data = pd.DataFrame({
            'close': np.random.uniform(50000, 51000, 5000),
            'high': np.random.uniform(50500, 51500, 5000),
            'low': np.random.uniform(49500, 50500, 5000),
            'open': np.random.uniform(50000, 51000, 5000),
            'volume': np.random.uniform(100, 1000, 5000)
        }, index=dates)

        class LargeDataFetcher:
            async def get_historical_data(self, symbol: str, timeframe: str, limit: int = 100):
                if len(large_data) > limit:
                    return large_data.tail(limit)
                return large_data

        # Configure strategy for large data
        config = StrategyConfig(
            name="Large Data RSI Strategy",
            symbols=["BTC/USDT"],
            timeframe="1h",
            required_history=100,  # Require more history
            params={
                "rsi_period": 14,
                "overbought": 70,
                "oversold": 30,
                "volume_threshold": 1.0
            }
        )

        fetcher = LargeDataFetcher()
        strategy = RSIStrategy(config)
        await strategy.initialize(fetcher)

        # Run strategy with large dataset
        signals = await strategy.run()

        # Verify strategy handled large data appropriately
        assert isinstance(signals, list)

    @pytest.mark.asyncio
    async def test_concurrent_strategy_execution(self):
        """
        Test concurrent execution of multiple strategies.

        Verifies that the system can handle multiple strategies
        running simultaneously without interference.
        """
        import asyncio

        async def run_strategy_instance(instance_id: int):
            """Run a single strategy instance."""
            config = StrategyConfig(
                name=f"Concurrent RSI Strategy {instance_id}",
                symbols=["BTC/USDT"],
                timeframe="1h",
                required_history=50,
                params={
                    "rsi_period": 14,
                    "overbought": 70,
                    "oversold": 30,
                    "volume_threshold": 1.0
                }
            )

            fetcher = MockDataFetcher()
            strategy = RSIStrategy(config)
            await strategy.initialize(fetcher)

            signals = await strategy.run()
            return len(signals)

        # Run multiple strategies concurrently
        tasks = [run_strategy_instance(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Verify all strategies completed successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, int)
            assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
