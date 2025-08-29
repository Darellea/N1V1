"""
tests/test_risk.py

Unit tests for the risk management system.
Tests position sizing, stop-loss/take-profit calculation, and risk rule enforcement.
"""

import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from typing import List
import numpy as np
import pandas as pd

from risk.risk_manager import RiskManager
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import OrderType


@pytest.fixture
def risk_config():
    """Fixture providing default risk configuration."""
    return {
        "stop_loss": 0.02,  # 2%
        "take_profit": 0.04,  # 4%
        "trailing_stop": True,
        "position_size": 0.1,  # 10%
        "max_position_size": 0.3,  # 30%
        "risk_reward_ratio": 2.0,
        "max_daily_drawdown": 0.1,  # 10%
        "require_stop_loss": True,
        "max_concurrent_trades": 3,
    }


@pytest.fixture
def risk_manager(risk_config):
    """Fixture providing initialized RiskManager instance."""
    return RiskManager(risk_config)


@pytest.fixture
def long_signal():
    """Fixture providing a long entry signal."""
    return TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type=OrderType.MARKET,
        amount=0,
        current_price=100.0,
        stop_loss=95.0,
    )


@pytest.fixture
def short_signal():
    """Fixture providing a short entry signal."""
    return TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_SHORT,
        signal_strength=SignalStrength.STRONG,
        order_type=OrderType.MARKET,
        amount=0,
        current_price=100.0,
        stop_loss=105.0,
    )


@pytest.mark.asyncio
async def test_initialization(risk_manager, risk_config):
    """Test RiskManager initialization with config."""
    assert risk_manager.require_stop_loss == risk_config["require_stop_loss"]
    assert risk_manager.max_position_size == Decimal(
        str(risk_config["max_position_size"])
    )
    assert risk_manager.max_daily_loss == Decimal(
        str(risk_config["max_daily_drawdown"])
    )
    assert risk_manager.risk_reward_ratio == Decimal(
        str(risk_config["risk_reward_ratio"])
    )


@pytest.mark.asyncio
async def test_signal_validation(risk_manager, long_signal):
    """Test basic signal validation."""
    # Valid signal should pass
    assert await risk_manager.evaluate_signal(long_signal) is True

    # Signal missing stop loss should fail when required
    invalid_signal = long_signal.copy()
    invalid_signal.stop_loss = None
    assert await risk_manager.evaluate_signal(invalid_signal) is False

    # Signal with invalid symbol should fail
    invalid_signal = long_signal.copy()
    invalid_signal.symbol = None
    assert await risk_manager.evaluate_signal(invalid_signal) is False


@pytest.mark.asyncio
async def test_fixed_fractional_position_sizing(risk_manager, long_signal):
    """Test fixed fractional position sizing."""
    # Mock current balance
    risk_manager._get_current_balance = AsyncMock(return_value=Decimal("10000"))

    # Test with stop loss
    position_size = await risk_manager.calculate_position_size(long_signal)
    expected_size = (
        Decimal("10000") * Decimal("0.1") / Decimal("0.05")
    )  # 5% risk (100->95)
    assert position_size == expected_size.quantize(Decimal(".000001"))

    # Test without stop loss (should use fixed %)
    no_sl_signal = long_signal.copy()
    no_sl_signal.stop_loss = None
    position_size = await risk_manager.calculate_position_size(no_sl_signal)
    expected_size = Decimal("10000") * Decimal("0.1")
    assert position_size == expected_size.quantize(Decimal(".000001"))


@pytest.mark.asyncio
async def test_volatility_position_sizing(risk_manager, long_signal):
    """Test volatility-based position sizing."""
    # Switch to volatility-based method
    risk_manager.position_sizing_method = "volatility"

    # Mock market data
    mock_data = {"close": pd.Series([90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])}

    # Mock current balance
    risk_manager._get_current_balance = AsyncMock(return_value=Decimal("10000"))

    # Test calculation
    position_size = await risk_manager.calculate_position_size(long_signal, mock_data)
    assert position_size > 0

    # Test with insufficient data (should fall back to fixed fractional)
    position_size = await risk_manager.calculate_position_size(
        long_signal, {"close": pd.Series([100])}
    )
    expected_size = Decimal("10000") * Decimal("0.1") / Decimal("0.05")
    assert position_size == expected_size.quantize(Decimal(".000001"))


@pytest.mark.asyncio
async def test_take_profit_calculation(risk_manager, long_signal, short_signal):
    """Test take profit calculation based on risk/reward ratio."""
    # Test long position
    tp = await risk_manager.calculate_take_profit(long_signal)
    expected_tp = 100 + (100 - 95) * 2  # 2.0 risk/reward ratio
    assert pytest.approx(tp) == expected_tp

    # Test short position
    tp = await risk_manager.calculate_take_profit(short_signal)
    expected_tp = 100 - (105 - 100) * 2  # 2.0 risk/reward ratio
    assert pytest.approx(tp) == expected_tp

    # Test without stop loss (should return None)
    no_sl_signal = long_signal.copy()
    no_sl_signal.stop_loss = None
    assert await risk_manager.calculate_take_profit(no_sl_signal) is None


@pytest.mark.asyncio
async def test_position_size_validation(risk_manager, long_signal):
    """Test position size validation against risk rules."""
    # Mock current balance
    risk_manager._get_current_balance = AsyncMock(return_value=Decimal("10000"))

    # Test valid position size
    long_signal.amount = 1000  # 10% of balance
    assert await risk_manager.evaluate_signal(long_signal) is True

    # Test position size exceeding max
    long_signal.amount = 4000  # 40% of balance (max is 30%)
    assert await risk_manager.evaluate_signal(long_signal) is False


@pytest.mark.asyncio
async def test_daily_loss_limit(risk_manager, long_signal):
    """Test daily loss limit enforcement."""
    # Set up loss condition
    risk_manager.today_start_balance = Decimal("10000")
    risk_manager.today_pnl = Decimal("-1100")  # -11% (limit is 10%)

    # Signal should be rejected
    assert await risk_manager.evaluate_signal(long_signal) is False


@pytest.mark.asyncio
async def test_max_concurrent_trades(risk_manager, long_signal):
    """Test maximum concurrent trades enforcement."""
    # Mock current positions
    risk_manager._get_current_positions = AsyncMock(
        return_value=["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    )

    # Signal should be rejected (already at max positions)
    assert await risk_manager.evaluate_signal(long_signal) is False


@pytest.mark.asyncio
async def test_emergency_check(risk_manager):
    """Test emergency market condition checks."""
    # Test normal conditions
    assert await risk_manager.emergency_check() is False

    # Test excessive loss condition (1.5x daily limit)
    risk_manager.today_start_balance = Decimal("10000")
    risk_manager.today_pnl = Decimal("-1500")  # -15% (1.5x 10% limit)
    assert await risk_manager.emergency_check() is True


@pytest.mark.asyncio
async def test_trade_outcome_updates(risk_manager):
    """Test updating risk models with trade outcomes."""
    # Initial state
    assert risk_manager.today_pnl == 0

    # Update with winning trade
    await risk_manager.update_trade_outcome("BTC/USDT", Decimal("100"), True)
    assert risk_manager.today_pnl == Decimal("100")

    # Update with losing trade
    await risk_manager.update_trade_outcome("BTC/USDT", Decimal("-50"), False)
    assert risk_manager.today_pnl == Decimal("50")


@pytest.mark.asyncio
async def test_volatility_tracking(risk_manager):
    """Test volatility tracking functionality."""
    # Test with sample price data
    prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 99, 100, 101])
    await risk_manager._update_volatility("BTC/USDT", prices)

    assert "BTC/USDT" in risk_manager.symbol_volatility
    assert risk_manager.symbol_volatility["BTC/USDT"]["volatility"] > 0


@pytest.mark.asyncio
async def test_risk_parameters_access(risk_manager):
    """Test getting current risk parameters."""
    params = await risk_manager.get_risk_parameters()
    assert "max_position_size" in params
    assert "risk_reward_ratio" in params
    assert "today_drawdown" in params

    # Test symbol-specific parameters
    params = await risk_manager.get_risk_parameters("BTC/USDT")
    assert "volatility" not in params  # No data yet

    # After updating volatility
    prices = pd.Series([100, 101, 102, 101, 100])
    await risk_manager._update_volatility("BTC/USDT", prices)
    params = await risk_manager.get_risk_parameters("BTC/USDT")
    assert "volatility" in params


def test_config_validation(risk_config):
    """Test that the test config matches the RiskManager expectations."""
    # Just validate that our fixture would pass RiskManager validation
    manager = RiskManager(risk_config)
    assert manager is not None
