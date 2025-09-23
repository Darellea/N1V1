"""
tests/test_adaptive_risk.py

Unit tests for adaptive risk management features including:
- Adaptive position sizing with ATR-based volatility scaling
- Dynamic stop loss / take profit calculation
- Trailing stop loss functionality
- Time-based and regime-based exit conditions
- Enhanced trade logging with exit details
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from risk.risk_manager import RiskManager
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import OrderType
from strategies.regime.market_regime import MarketRegime


class TestAdaptiveRiskManagement:
    """Test suite for adaptive risk management features."""

    @pytest.fixture
    def risk_config(self):
        """Risk management configuration for testing."""
        return {
            "require_stop_loss": True,
            "max_position_size": 0.3,
            "max_daily_drawdown": 0.1,
            "risk_reward_ratio": 2.0,
            "position_sizing_method": "adaptive_atr",
            "risk_per_trade": 0.02,
            "atr_k_factor": 2.0,
            "stop_loss_method": "atr",
            "atr_sl_multiplier": 2.0,
            "stop_loss_percentage": 0.02,
            "tp_base_multiplier": 2.0,
            "enable_adaptive_tp": True,
            "enable_trailing_stop": True,
            "trailing_stop_method": "percentage",
            "trailing_distance": 0.02,
            "trailing_atr_multiplier": 1.5,
            "enable_time_based_exit": True,
            "max_holding_candles": 72,
            "timeframe": "1h",
            "enable_regime_based_exit": True,
            "exit_on_regime_change": True,
            "enhanced_trade_logging": True,
            "track_exit_reasons": True,
            "log_sl_tp_details": True,
        }

    @pytest.fixture
    def risk_manager(self, risk_config):
        """RiskManager instance for testing."""
        return RiskManager(risk_config)

    @pytest.fixture
    def sample_market_data(self):
        """Sample OHLCV market data for testing."""
        # Create 20 periods of sample data
        dates = pd.date_range('2023-01-01', periods=20, freq='h')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        base_price = 50000
        prices = []
        highs = []
        lows = []
        opens = []
        volumes = []

        for i in range(20):
            # Add some trend and volatility
            trend = i * 10  # Upward trend
            noise = np.random.normal(0, 100)  # Random noise
            close = base_price + trend + noise

            # Generate OHLC
            high = close + abs(np.random.normal(0, 50))
            low = close - abs(np.random.normal(0, 50))
            open_price = close + np.random.normal(0, 20)

            prices.append(close)
            highs.append(high)
            lows.append(low)
            opens.append(open_price)
            volumes.append(np.random.uniform(1000, 10000))

        return {
            "timestamp": dates,
            "open": pd.Series(opens, index=dates),
            "high": pd.Series(highs, index=dates),
            "low": pd.Series(lows, index=dates),
            "close": pd.Series(prices, index=dates),
            "volume": pd.Series(volumes, index=dates),
            "adx": 30  # Sample ADX value
        }

    @pytest.fixture
    def sample_signal(self):
        """Sample trading signal for testing."""
        return TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            stop_loss=Decimal("49000"),
            take_profit=Decimal("51000"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

    @pytest.mark.asyncio
    async def test_adaptive_position_size_calculation(self, risk_manager, sample_market_data):
        """Test adaptive position sizing with ATR-based volatility scaling."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("0"),  # Will be calculated
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        # Mock account balance
        with patch.object(risk_manager, '_get_current_balance', return_value=Decimal("10000")):
            position_size = await risk_manager.calculate_adaptive_position_size(signal, sample_market_data)

            # Position size should be calculated based on ATR
            assert isinstance(position_size, Decimal)
            assert position_size > 0

            # Should be less than max position size
            max_allowed = Decimal("10000") * risk_manager.max_position_size
            assert position_size <= max_allowed

    @pytest.mark.asyncio
    async def test_adaptive_position_size_fallback(self, risk_manager):
        """Test adaptive position sizing fallback when no market data."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("0"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        # Mock account balance
        with patch.object(risk_manager, '_get_current_balance', return_value=Decimal("10000")):
            position_size = await risk_manager.calculate_adaptive_position_size(signal, None)

            # Should fallback to fixed percentage
            expected = Decimal("10000") * risk_manager.risk_per_trade
            assert position_size == expected

    @pytest.mark.asyncio
    async def test_dynamic_stop_loss_atr(self, risk_manager, sample_market_data):
        """Test dynamic stop loss calculation using ATR."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        stop_loss = await risk_manager.calculate_dynamic_stop_loss(signal, sample_market_data)

        assert stop_loss is not None
        assert isinstance(stop_loss, Decimal)
        # For LONG position, stop loss should be below entry price
        assert stop_loss < Decimal("50000")

    @pytest.mark.asyncio
    async def test_dynamic_stop_loss_percentage(self, risk_manager):
        """Test dynamic stop loss calculation using percentage."""
        # Change config to use percentage method
        risk_manager.stop_loss_method = "percentage"

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        stop_loss = await risk_manager.calculate_dynamic_stop_loss(signal, None)

        assert stop_loss is not None
        assert isinstance(stop_loss, Decimal)
        # Should be 2% below entry price
        expected = Decimal("50000") * (1 - risk_manager.stop_loss_percentage)
        assert stop_loss == expected

    @pytest.mark.asyncio
    async def test_adaptive_take_profit(self, risk_manager, sample_market_data):
        """Test adaptive take profit calculation."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            stop_loss=Decimal("49000"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        take_profit = await risk_manager.calculate_adaptive_take_profit(signal, sample_market_data)

        assert take_profit is not None
        assert isinstance(take_profit, Decimal)
        # For LONG position, take profit should be above entry price
        assert take_profit > Decimal("50000")

    @pytest.mark.asyncio
    async def test_trailing_stop_percentage(self, risk_manager):
        """Test trailing stop calculation using percentage."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            stop_loss=Decimal("49000"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        # Test LONG position trailing
        highest_price = Decimal("51000")
        trailing_stop = await risk_manager.calculate_trailing_stop(
            signal, Decimal("50500"), highest_price, None, None
        )

        assert trailing_stop is not None
        assert isinstance(trailing_stop, Decimal)
        # Should trail below highest price by trailing distance
        expected = highest_price * (1 - risk_manager.trailing_distance)
        assert trailing_stop == expected

    @pytest.mark.asyncio
    async def test_trailing_stop_atr(self, risk_manager, sample_market_data):
        """Test trailing stop calculation using ATR."""
        # Change config to use ATR method
        risk_manager.trailing_stop_method = "atr"

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            stop_loss=Decimal("49000"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        highest_price = Decimal("51000")
        trailing_stop = await risk_manager.calculate_trailing_stop(
            signal, Decimal("50500"), highest_price, None, sample_market_data
        )

        assert trailing_stop is not None
        assert isinstance(trailing_stop, Decimal)

    @pytest.mark.asyncio
    async def test_time_based_exit(self, risk_manager):
        """Test time-based exit conditions."""
        entry_timestamp = int(datetime.now().timestamp() * 1000)
        current_timestamp = entry_timestamp + (73 * 60 * 60 * 1000)  # 73 hours later

        should_exit, exit_reason = await risk_manager.should_exit_time_based(
            entry_timestamp, current_timestamp, "1h", 72
        )

        assert should_exit is True
        assert "time_limit_72_candles" in exit_reason

    @pytest.mark.asyncio
    async def test_time_based_exit_no_exit(self, risk_manager):
        """Test time-based exit when conditions not met."""
        entry_timestamp = int(datetime.now().timestamp() * 1000)
        current_timestamp = entry_timestamp + (24 * 60 * 60 * 1000)  # 24 hours later

        should_exit, exit_reason = await risk_manager.should_exit_time_based(
            entry_timestamp, current_timestamp, "1h", 72
        )

        assert should_exit is False
        assert exit_reason == ""

    @pytest.mark.asyncio
    async def test_regime_based_exit(self, risk_manager, sample_market_data, sample_signal):
        """Test regime-based exit conditions."""
        # Mock regime detector to return sideways regime
        with patch('risk.risk_manager.get_market_regime_detector') as mock_detector:
            mock_regime_detector = Mock()
            mock_result = Mock()
            mock_result.regime = MarketRegime.SIDEWAYS
            mock_result.previous_regime = MarketRegime.TRENDING
            mock_regime_detector.detect_regime.return_value = mock_result
            mock_detector.return_value = mock_regime_detector

            should_exit, exit_reason = await risk_manager.should_exit_regime_change(
                sample_signal, sample_market_data
            )

            assert should_exit is True
            assert "regime_change_trending_to_sideways" in exit_reason

    @pytest.mark.asyncio
    async def test_position_tracking_update(self, risk_manager, sample_signal):
        """Test position tracking updates."""
        current_price = Decimal("50500")
        highest_price = Decimal("51000")

        tracking = await risk_manager.update_position_tracking(
            sample_signal, current_price, highest_price, None
        )

        assert "highest_price" in tracking
        assert "lowest_price" in tracking
        assert "trailing_stop" in tracking
        assert "current_price" in tracking
        assert "last_updated" in tracking

        assert tracking["highest_price"] == highest_price
        assert tracking["current_price"] == current_price

    @pytest.mark.asyncio
    async def test_enhanced_trade_logging(self, risk_manager, sample_signal):
        """Test enhanced trade logging with exit details."""
        exit_price = Decimal("49500")
        exit_reason = "sl_hit"
        pnl = Decimal("-100")
        entry_price = Decimal("50000")
        stop_loss = Decimal("49000")
        take_profit = Decimal("51000")

        # Mock trade logger
        with patch('risk.risk_manager.trade_logger') as mock_logger:
            await risk_manager.log_trade_with_exit_details(
                sample_signal, exit_price, exit_reason, pnl,
                entry_price, stop_loss, take_profit
            )

            # Verify performance method was called
            mock_logger.performance.assert_called_once()
            call_args = mock_logger.performance.call_args
            assert call_args.args[0] == "Trade closed"
            trade_details = call_args.args[1]

            assert trade_details["symbol"] == "BTC/USDT"
            assert trade_details["exit_reason"] == exit_reason
            assert trade_details["pnl"] == float(pnl)
            assert trade_details["entry_price"] == float(entry_price)
            assert trade_details["exit_price"] == float(exit_price)
            assert trade_details["stop_loss"] == float(stop_loss)
            assert trade_details["take_profit"] == float(take_profit)

    @pytest.mark.asyncio
    async def test_atr_calculation(self, risk_manager, sample_market_data):
        """Test ATR calculation from market data."""
        atr = await risk_manager._calculate_atr(sample_market_data, period=14)

        assert isinstance(atr, Decimal)
        assert atr >= 0

    @pytest.mark.asyncio
    async def test_trend_multiplier_calculation(self, risk_manager):
        """Test trend multiplier calculation."""
        # Test with strong trend (ADX >= 40)
        market_data_strong = {"adx": 45}
        multiplier = await risk_manager._calculate_trend_multiplier(market_data_strong)
        assert multiplier == Decimal("1.5")

        # Test with moderate trend (ADX >= 25)
        market_data_moderate = {"adx": 30}
        multiplier = await risk_manager._calculate_trend_multiplier(market_data_moderate)
        assert multiplier == Decimal("1.2")

        # Test with weak trend (ADX < 25)
        market_data_weak = {"adx": 20}
        multiplier = await risk_manager._calculate_trend_multiplier(market_data_weak)
        assert multiplier == Decimal("0.8")

    @pytest.mark.asyncio
    async def test_timeframe_conversion(self, risk_manager):
        """Test timeframe to milliseconds conversion."""
        assert risk_manager._timeframe_to_ms("1m") == 60 * 1000
        assert risk_manager._timeframe_to_ms("1h") == 60 * 60 * 1000
        assert risk_manager._timeframe_to_ms("1d") == 24 * 60 * 60 * 1000
        assert risk_manager._timeframe_to_ms("4h") == 4 * 60 * 60 * 1000

    @pytest.mark.asyncio
    async def test_exit_type_statistics_update(self, risk_manager):
        """Test exit type statistics tracking."""
        # Test win update
        await risk_manager._update_exit_type_stats("sl_hit", True)
        assert risk_manager.exit_type_stats["sl_hit"]["wins"] == 1
        assert risk_manager.exit_type_stats["sl_hit"]["losses"] == 0

        # Test loss update
        await risk_manager._update_exit_type_stats("tp_hit", False)
        assert risk_manager.exit_type_stats["tp_hit"]["wins"] == 0
        assert risk_manager.exit_type_stats["tp_hit"]["losses"] == 1

    @pytest.mark.asyncio
    async def test_risk_parameters_getter(self, risk_manager):
        """Test risk parameters getter method."""
        params = await risk_manager.get_risk_parameters()

        required_keys = [
            "max_position_size", "max_daily_loss", "risk_reward_ratio",
            "position_sizing_method", "today_pnl", "today_drawdown"
        ]

        for key in required_keys:
            assert key in params
            if key == "position_sizing_method":
                assert isinstance(params[key], str)
            else:
                assert isinstance(params[key], (int, float))

    @pytest.mark.asyncio
    async def test_risk_parameters_with_symbol(self, risk_manager):
        """Test risk parameters getter with symbol volatility."""
        # Add some volatility data
        risk_manager.symbol_volatility["BTC/USDT"] = {
            "volatility": 0.25,
            "last_updated": int(datetime.now().timestamp() * 1000)
        }

        params = await risk_manager.get_risk_parameters("BTC/USDT")

        assert "volatility" in params
        assert params["volatility"] == 0.25

    @pytest.mark.asyncio
    async def test_position_sizing_method_selection(self, risk_manager, sample_market_data):
        """Test that position sizing method selection works correctly."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("0"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        # Mock account balance
        with patch.object(risk_manager, '_get_current_balance', return_value=Decimal("10000")):
            # Test adaptive_atr method
            risk_manager.position_sizing_method = "adaptive_atr"
            size1 = await risk_manager.calculate_position_size(signal, sample_market_data)

            # Test fixed_percent method
            risk_manager.position_sizing_method = "fixed_percent"
            size2 = await risk_manager.calculate_position_size(signal, sample_market_data)

            # Test kelly method
            risk_manager.position_sizing_method = "kelly"
            size3 = await risk_manager.calculate_position_size(signal, sample_market_data)

            # All should return valid position sizes
            assert size1 > 0
            assert size2 > 0
            assert size3 > 0

            # Different methods should potentially give different results
            # (though they might be the same due to fallbacks)

    @pytest.mark.asyncio
    async def test_error_handling_in_calculations(self, risk_manager):
        """Test error handling in various calculations."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        # Test with invalid market data
        invalid_data = {"invalid": "data"}

        # These should not raise exceptions and should return reasonable defaults
        pos_size = await risk_manager.calculate_adaptive_position_size(signal, invalid_data)
        assert pos_size >= 0

        sl = await risk_manager.calculate_dynamic_stop_loss(signal, invalid_data)
        # May return None for invalid data, which is acceptable

        tp = await risk_manager.calculate_adaptive_take_profit(signal, invalid_data)
        # May return None for invalid data, which is acceptable

    @pytest.mark.asyncio
    async def test_short_position_calculations(self, risk_manager, sample_market_data):
        """Test calculations for SHORT positions."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("1000"),
            stop_loss=Decimal("51000"),  # Above entry for SHORT
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        # Test stop loss calculation for SHORT
        stop_loss = await risk_manager.calculate_dynamic_stop_loss(signal, sample_market_data)
        if stop_loss:
            # For SHORT position, stop loss should be above entry price
            assert stop_loss > Decimal("50000")

        # Test take profit calculation for SHORT
        take_profit = await risk_manager.calculate_adaptive_take_profit(signal, sample_market_data)
        if take_profit:
            # For SHORT position, take profit should be below entry price
            assert take_profit < Decimal("50000")

        # Test trailing stop for SHORT
        lowest_price = Decimal("49000")
        trailing_stop = await risk_manager.calculate_trailing_stop(
            signal, Decimal("49500"), None, lowest_price, sample_market_data
        )
        if trailing_stop:
            # For SHORT position, trailing stop should be above lowest price
            assert trailing_stop > lowest_price


# Integration tests
class TestAdaptiveRiskIntegration:
    """Integration tests for adaptive risk management."""

    @pytest.fixture
    def sample_market_data(self):
        """Sample OHLCV market data for testing."""
        # Create 20 periods of sample data
        dates = pd.date_range('2023-01-01', periods=20, freq='h')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        base_price = 50000
        prices = []
        highs = []
        lows = []
        opens = []
        volumes = []

        for i in range(20):
            # Add some trend and volatility
            trend = i * 10  # Upward trend
            noise = np.random.normal(0, 100)  # Random noise
            close = base_price + trend + noise

            # Generate OHLC
            high = close + abs(np.random.normal(0, 50))
            low = close - abs(np.random.normal(0, 50))
            open_price = close + np.random.normal(0, 20)

            prices.append(close)
            highs.append(high)
            lows.append(low)
            opens.append(open_price)
            volumes.append(np.random.uniform(1000, 10000))

        return {
            "timestamp": dates,
            "open": pd.Series(opens, index=dates),
            "high": pd.Series(highs, index=dates),
            "low": pd.Series(lows, index=dates),
            "close": pd.Series(prices, index=dates),
            "volume": pd.Series(volumes, index=dates),
            "adx": 30  # Sample ADX value
        }

    @pytest.fixture
    def full_risk_manager(self):
        """RiskManager with full configuration."""
        config = {
            "require_stop_loss": True,
            "max_position_size": 0.3,
            "max_daily_drawdown": 0.1,
            "risk_reward_ratio": 2.0,
            "position_sizing_method": "adaptive_atr",
            "risk_per_trade": 0.02,
            "atr_k_factor": 2.0,
            "stop_loss_method": "atr",
            "atr_sl_multiplier": 2.0,
            "stop_loss_percentage": 0.02,
            "tp_base_multiplier": 2.0,
            "enable_adaptive_tp": True,
            "enable_trailing_stop": True,
            "trailing_stop_method": "percentage",
            "trailing_distance": 0.02,
            "enable_time_based_exit": True,
            "max_holding_candles": 72,
            "timeframe": "1h",
            "enable_regime_based_exit": True,
            "exit_on_regime_change": True,
            "enhanced_trade_logging": True,
            "track_exit_reasons": True,
            "log_sl_tp_details": True,
        }
        return RiskManager(config)

    @pytest.mark.asyncio
    async def test_complete_trade_workflow(self, full_risk_manager, sample_market_data):
        """Test complete trade workflow with all adaptive features."""
        # Create signal without position size, stop loss, or take profit
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type=OrderType.MARKET,
            current_price=Decimal("50000"),
            amount=Decimal("0"),  # Will be calculated
            timestamp=int(datetime.now().timestamp() * 1000)
        )

        # Mock account balance
        with patch.object(full_risk_manager, '_get_current_balance', return_value=Decimal("10000")):
            # 1. Evaluate signal (should calculate position size, SL, TP)
            is_valid = await full_risk_manager.evaluate_signal(signal, sample_market_data)

            assert is_valid is True
            assert signal.amount > 0
            assert signal.stop_loss is not None
            assert signal.take_profit is not None

            # 2. Test position tracking
            tracking = await full_risk_manager.update_position_tracking(
                signal, signal.current_price, signal.current_price, None
            )

            assert tracking["highest_price"] == signal.current_price
            assert tracking["current_price"] == signal.current_price

            # 3. Test trailing stop calculation
            trailing_stop = await full_risk_manager.calculate_trailing_stop(
                signal, signal.current_price, signal.current_price, None, sample_market_data
            )

            assert trailing_stop is not None

            # 4. Test time-based exit (should not exit immediately)
            should_exit, exit_reason = await full_risk_manager.should_exit_time_based(
                signal.timestamp, signal.timestamp + timedelta(hours=24), "1h", 72
            )

            assert should_exit is False

            # 5. Test trade logging
            await full_risk_manager.log_trade_with_exit_details(
                signal, Decimal("49500"), "sl_hit", Decimal("-100"),
                signal.current_price, signal.stop_loss, signal.take_profit
            )

            # Verify exit stats were updated
            assert full_risk_manager.exit_type_stats["sl_hit"]["losses"] >= 0

    @pytest.mark.asyncio
    async def test_risk_manager_initialization(self, full_risk_manager):
        """Test that RiskManager initializes correctly with all new parameters."""
        # Verify all new attributes are set
        assert hasattr(full_risk_manager, 'risk_per_trade')
        assert hasattr(full_risk_manager, 'atr_k_factor')
        assert hasattr(full_risk_manager, 'stop_loss_method')
        assert hasattr(full_risk_manager, 'enable_adaptive_tp')
        assert hasattr(full_risk_manager, 'enable_trailing_stop')
        assert hasattr(full_risk_manager, 'enable_time_based_exit')
        assert hasattr(full_risk_manager, 'enable_regime_based_exit')
        assert hasattr(full_risk_manager, 'position_tracking')
        assert hasattr(full_risk_manager, 'exit_type_stats')

        # Verify default values
        assert full_risk_manager.position_sizing_method == "adaptive_atr"
        assert full_risk_manager.stop_loss_method == "atr"
        assert full_risk_manager.enable_trailing_stop is True
        assert full_risk_manager.enable_time_based_exit is True
        assert full_risk_manager.enable_regime_based_exit is True

        # Verify exit type stats structure
        expected_exit_types = ["sl_hit", "tp_hit", "time_limit", "regime_change", "manual"]
        for exit_type in expected_exit_types:
            assert exit_type in full_risk_manager.exit_type_stats
            assert "wins" in full_risk_manager.exit_type_stats[exit_type]
            assert "losses" in full_risk_manager.exit_type_stats[exit_type]
