import asyncio
from decimal import Decimal
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd

from core.signal_router import SignalRouter
from core.contracts import TradingSignal, SignalType, SignalStrength


class DummyRiskManager:
    """Simple risk manager stub that approves all signals."""

    require_stop_loss = False

    async def evaluate_signal(self, signal, market_data=None):
        await asyncio.sleep(0)  # yield control to allow concurrency in tests
        return True


@pytest.mark.asyncio
async def test_ml_confirmation_rejects_weak_signal_on_opposite_prediction():
    """Test that ML confirmation rejects a weak signal when ML predicts opposite direction with high confidence."""

    # Create router with ML enabled
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Manually enable ML for this test
    router.ml_enabled = True
    router.ml_confidence_threshold = 0.6

    # Create a mock ML filter that rejects the signal
    mock_filter = MagicMock()
    mock_filter.filter_signal.return_value = {
        'approved': False,
        'confidence': 0.8,
        'reason': 'direction_mismatch'
    }
    router.ml_filter = mock_filter

    # Create a weak BUY signal (ENTRY_LONG)
    weak_buy_signal = TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,  # BUY signal
        signal_strength=SignalStrength.WEAK,  # Weak strength
        order_type="market",
        amount=Decimal("1.0"),
        price=Decimal("50000"),
        current_price=Decimal("50000"),
        timestamp=1234567890,
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
    )

    # Mock market data with features
    market_data = {
        "features": pd.DataFrame([{"feature1": 1.0, "feature2": 2.0}])
    }

    result = await router.process_signal(weak_buy_signal, market_data)

    # Signal should be rejected because ML filter returned approved=False
    assert result is None
    assert len(router.get_active_signals()) == 0
    # Verify the ML filter was called
    mock_filter.filter_signal.assert_called_once()


@pytest.mark.asyncio
async def test_ml_confirmation_accepts_signal_on_matching_prediction():
    """Test that ML confirmation accepts a signal when ML predicts the same direction with high confidence."""

    # Create a mock ML model and prediction function
    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": 1, "confidence": 0.8}])  # Same direction, high confidence

    def mock_predict(model, features_df):
        return mock_prediction_df

    # Create router with ML enabled
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Manually enable ML for this test
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    # Create a BUY signal (ENTRY_LONG)
    buy_signal = TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,  # BUY signal
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        price=Decimal("50000"),
        current_price=Decimal("50000"),
        timestamp=1234567890,
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
    )

    # Mock market data with features
    market_data = {
        "features": pd.DataFrame([{"feature1": 1.0, "feature2": 2.0}])
    }

    # Mock the ML prediction function
    with patch('core.signal_router.ml_predict', side_effect=mock_predict):
        result = await router.process_signal(buy_signal, market_data)

    # Signal should be accepted because ML prediction matches signal direction
    assert result is not None
    assert result.symbol == "BTC/USDT"
    assert len(router.get_active_signals()) == 1


@pytest.mark.asyncio
async def test_ml_confirmation_skips_when_no_features():
    """Test that ML confirmation is skipped when no feature data is available."""

    # Create router with ML enabled
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Manually enable ML for this test
    router.ml_enabled = True
    router.ml_model = MagicMock()
    router.ml_confidence_threshold = 0.6

    # Create a BUY signal (ENTRY_LONG)
    buy_signal = TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        price=Decimal("50000"),
        current_price=Decimal("50000"),
        timestamp=1234567890,
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
    )

    # Mock market data without features
    market_data = {
        "ohlcv": [100, 105, 95, 102, 1000]  # No features key
    }

    # Process signal without mocking ML (should skip ML confirmation)
    result = await router.process_signal(buy_signal, market_data)

    # Signal should be accepted because ML confirmation is skipped
    assert result is not None
    assert result.symbol == "BTC/USDT"
    assert len(router.get_active_signals()) == 1
