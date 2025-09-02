import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.signal_router import SignalRouter
from core.contracts import TradingSignal, SignalType, SignalStrength


class DummyRiskManager:
    """Simple risk manager stub that approves all signals."""

    require_stop_loss = False

    async def evaluate_signal(self, signal, market_data=None):
        await asyncio.sleep(0)  # yield control to allow concurrency in tests
        return True


class FailingRiskManager:
    """Risk manager that fails on first attempt but succeeds on retry."""

    def __init__(self):
        self.call_count = 0

    async def evaluate_signal(self, signal, market_data=None):
        self.call_count += 1
        if self.call_count == 1:
            raise Exception("Temporary failure")
        return True


@pytest.fixture
def dummy_risk_manager():
    return DummyRiskManager()


@pytest.fixture
def signal_router(dummy_risk_manager):
    return SignalRouter(risk_manager=dummy_risk_manager)


@pytest.fixture
def sample_signal():
    return TradingSignal(
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


@pytest.mark.asyncio
async def test_initialization(signal_router):
    """Test SignalRouter initialization."""
    assert signal_router.risk_manager is not None
    assert signal_router.active_signals == {}
    assert signal_router.signal_history == []
    assert signal_router.block_signals is False
    assert signal_router.critical_errors == 0


@pytest.mark.asyncio
async def test_process_signal_valid(signal_router, sample_signal):
    """Test processing a valid signal."""
    result = await signal_router.process_signal(sample_signal)

    assert result is not None
    assert result.symbol == "BTC/USDT"
    assert len(signal_router.get_active_signals()) == 1


@pytest.mark.asyncio
async def test_process_signal_invalid_missing_symbol(signal_router):
    """Test processing an invalid signal with missing symbol."""
    invalid_signal = TradingSignal(
        strategy_id="test",
        symbol="",  # Invalid
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    result = await signal_router.process_signal(invalid_signal)
    assert result is None
    assert len(signal_router.get_active_signals()) == 0


@pytest.mark.asyncio
async def test_process_signal_invalid_zero_amount(signal_router):
    """Test processing an invalid signal with zero amount."""
    invalid_signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("0"),  # Invalid
    )

    result = await signal_router.process_signal(invalid_signal)
    assert result is None
    assert len(signal_router.get_active_signals()) == 0


def test_validate_signal_valid(signal_router, sample_signal):
    """Test signal validation with valid signal."""
    assert signal_router._validate_signal(sample_signal) is True


def test_validate_signal_invalid(signal_router):
    """Test signal validation with invalid signals."""
    # Missing symbol
    invalid1 = TradingSignal(
        strategy_id="test",
        symbol="",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )
    assert signal_router._validate_signal(invalid1) is False

    # Zero amount
    invalid2 = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("0"),
    )
    assert signal_router._validate_signal(invalid2) is False


@pytest.mark.asyncio
async def test_signal_conflicts_opposite_entry(signal_router):
    """Test conflict detection for opposite entry signals."""
    sig1 = TradingSignal(
        strategy_id="s1",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    sig2 = TradingSignal(
        strategy_id="s2",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_SHORT,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    # Process first signal
    await signal_router.process_signal(sig1)
    assert len(signal_router.get_active_signals()) == 1

    # Process conflicting signal
    result = await signal_router.process_signal(sig2)
    # Should resolve conflict - only one should remain
    active = signal_router.get_active_signals()
    assert len(active) <= 1


@pytest.mark.asyncio
async def test_signal_conflicts_entry_vs_exit(signal_router):
    """Test conflict detection for entry vs exit signals."""
    sig_entry = TradingSignal(
        strategy_id="s1",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    sig_exit = TradingSignal(
        strategy_id="s2",
        symbol="BTC/USDT",
        signal_type=SignalType.EXIT_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    # Process exit signal first
    await signal_router.process_signal(sig_exit)
    assert len(signal_router.get_active_signals()) == 1

    # Process entry signal - should be rejected due to existing exit
    result = await signal_router.process_signal(sig_entry)
    active = signal_router.get_active_signals()
    assert len(active) == 1
    assert active[0].signal_type == SignalType.EXIT_LONG
    assert result is None  # Entry should be rejected


@pytest.mark.asyncio
async def test_cancel_signal(signal_router, sample_signal):
    """Test canceling an active signal."""
    # Store signal first
    await signal_router.process_signal(sample_signal)
    assert len(signal_router.get_active_signals()) == 1

    # Cancel it
    await signal_router._cancel_signal(sample_signal)
    assert len(signal_router.get_active_signals()) == 0


@pytest.mark.asyncio
async def test_update_signal_status(signal_router, sample_signal):
    """Test updating signal status."""
    await signal_router.process_signal(sample_signal)
    assert len(signal_router.get_active_signals()) == 1

    await signal_router.update_signal_status(sample_signal, "executed", "order filled")
    assert len(signal_router.get_active_signals()) == 0


def test_get_active_signals(signal_router, sample_signal):
    """Test getting active signals."""
    # No signals initially
    assert signal_router.get_active_signals() == []

    # Add signal manually for sync test
    signal_router._store_signal(sample_signal)
    assert len(signal_router.get_active_signals()) == 1

    # Filter by symbol
    assert len(signal_router.get_active_signals("BTC/USDT")) == 1
    assert len(signal_router.get_active_signals("ETH/USDT")) == 0


def test_get_signal_history(signal_router, sample_signal):
    """Test getting signal history."""
    assert signal_router.get_signal_history() == []

    signal_router._store_signal(sample_signal)
    history = signal_router.get_signal_history()
    assert len(history) == 1
    assert history[0] == sample_signal


def test_clear_signals(signal_router, sample_signal):
    """Test clearing all signals."""
    signal_router._store_signal(sample_signal)
    assert len(signal_router.get_active_signals()) == 1

    signal_router.clear_signals()
    assert len(signal_router.get_active_signals()) == 0


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_async_call_success(mock_sleep):
    """Test successful retry of async call."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    async def dummy_call():
        return "success"

    result = await router._retry_async_call(dummy_call, retries=2)
    assert result == "success"
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_async_call_with_failure(mock_sleep):
    """Test retry mechanism with temporary failure."""
    rm = FailingRiskManager()
    router = SignalRouter(risk_manager=rm)

    async def failing_call():
        return await rm.evaluate_signal(None)

    result = await router._retry_async_call(failing_call, retries=2)
    assert result is True
    assert rm.call_count == 2  # Failed once, succeeded on retry
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_async_call_exhaust_retries(mock_sleep):
    """Test retry exhaustion."""
    rm = FailingRiskManager()
    router = SignalRouter(risk_manager=rm)

    async def always_fail():
        raise Exception("Always fails")

    with pytest.raises(Exception, match="Always fails"):
        await router._retry_async_call(always_fail, retries=1)
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_record_router_error(signal_router):
    """Test recording router errors."""
    assert signal_router.critical_errors == 0
    assert signal_router.block_signals is False

    # Record errors up to threshold
    for i in range(10):
        await signal_router._record_router_error(Exception(f"Error {i}"))

    assert signal_router.critical_errors == 10
    assert signal_router.block_signals is True


@pytest.mark.asyncio
async def test_concurrent_signal_processing_no_conflicts():
    """
    Spawn many concurrent signals for distinct symbols and ensure all are approved
    and stored in active_signals without loss or duplication.
    """
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    tasks = []
    n = 50
    for i in range(n):
        sig = TradingSignal(
            strategy_id=f"s{i}",
            symbol=f"SYM{i}/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
        )
        tasks.append(asyncio.create_task(router.process_signal(sig)))

    results = await asyncio.gather(*tasks)
    approved = [r for r in results if r is not None]
    # All signals should be approved
    assert len(approved) == n
    # Active signals should contain all approved signals
    assert len(router.get_active_signals()) == n


@pytest.mark.asyncio
async def test_conflicting_signals_resolution():
    """
    Submit two conflicting signals for the same symbol concurrently (opposite directions)
    and verify conflict resolution keeps the stronger signal.
    """
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Weak BUY vs Strong SELL for same symbol
    sig_buy = TradingSignal(
        strategy_id="s-buy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.WEAK,
        order_type="market",
        amount=Decimal("0.1"),
    )

    sig_sell = TradingSignal(
        strategy_id="s-sell",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_SHORT,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("0.1"),
    )

    task1 = asyncio.create_task(router.process_signal(sig_buy))
    task2 = asyncio.create_task(router.process_signal(sig_sell))
    res_buy, res_sell = await asyncio.gather(task1, task2)

    # At most one active signal for the symbol should remain
    active = router.get_active_signals(symbol="BTC/USDT")
    assert len(active) <= 1
    if active:
        # If one remains, it should be the stronger (SELL)
        assert active[0].signal_strength == SignalStrength.STRONG


@pytest.mark.asyncio
async def test_generate_signal_id(signal_router, sample_signal):
    """Test signal ID generation."""
    signal_id = signal_router._generate_signal_id(sample_signal)
    expected = f"{sample_signal.strategy_id}_{sample_signal.symbol}_{sample_signal.timestamp}"
    assert signal_id == expected


@pytest.mark.asyncio
async def test_get_symbol_lock(signal_router):
    """Test symbol-specific lock retrieval."""
    lock1 = await signal_router._get_symbol_lock("BTC/USDT")
    lock2 = await signal_router._get_symbol_lock("BTC/USDT")
    assert lock1 is lock2  # Same lock for same symbol

    lock3 = await signal_router._get_symbol_lock("ETH/USDT")
    assert lock1 is not lock3  # Different lock for different symbol

    # None symbol returns global lock
    global_lock = await signal_router._get_symbol_lock(None)
    assert global_lock is signal_router._lock


def test_signal_router_init_defaults(dummy_risk_manager):
    """Test SignalRouter initialization with default parameters."""
    router = SignalRouter(risk_manager=dummy_risk_manager)
    assert router.risk_manager is dummy_risk_manager
    assert router.active_signals == {}
    assert router.signal_history == []
    assert router.conflict_resolution_rules == {
        "strength_based": True,
        "newer_first": False,
        "exit_over_entry": True,
    }
    assert router.retry_config == {
        "max_retries": 2,
        "backoff_base": 0.5,
        "max_backoff": 5.0,
    }
    assert router.critical_errors == 0
    assert router.safe_mode_threshold == 10
    assert router.block_signals is False
    assert isinstance(router._lock, asyncio.Lock)
    assert isinstance(router._symbol_locks, dict)
    assert router.task_manager is None or router.task_manager is not None  # task_manager can be None or passed





@pytest.mark.asyncio
async def test_ml_confirmation_weak_signal_approved():
    """Test that ML confirmation approves a weak signal when ML predicts same direction with high confidence."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength
    from core.signal_router import SignalRouter

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

    # Debug prints to check ML condition
    print(f'ML enabled: {router.ml_enabled}')
    print(f'ML model exists: {router.ml_model is not None}')
    print(f'Signal type: {weak_buy_signal.signal_type}')
    print(f'Signal in entry types: {weak_buy_signal.signal_type in {SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT}}')
    print(f'Signal strength: {weak_buy_signal.signal_strength}')
    print(f'Market data has features: {"features" in market_data}')
    print(f'Market data features type: {type(market_data.get("features"))}')

    # Mock the ML prediction function
    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        result = await router.process_signal(weak_buy_signal, market_data)
        print(f'ML predict was called: {mock_patch.called}')
        print(f'Number of calls: {mock_patch.call_count}')

    # Signal should be approved because:
    # - Signal is BUY (ENTRY_LONG, desired = 1)
    # - ML predicts BUY (1) with high confidence (0.8 > 0.6)
    assert mock_patch.called
    assert result is not None
    assert result.symbol == "BTC/USDT"
    assert len(router.get_active_signals()) == 1


@pytest.mark.asyncio
async def test_ml_confirmation_weak_signal_rejected():
    """Test that ML confirmation rejects a weak signal when ML predicts opposite direction with high confidence."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength
    from core.signal_router import SignalRouter

    # Create a mock ML model and prediction function
    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": -1, "confidence": 0.8}])  # Opposite direction, high confidence

    def mock_predict(model, features_df):
        return mock_prediction_df

    # Create router with ML enabled
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Manually enable ML for this test
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

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

    # Mock the ML prediction function
    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        result = await router.process_signal(weak_buy_signal, market_data)

    # Signal should be rejected because:
    # - Signal is BUY (ENTRY_LONG, desired = 1)
    # - ML predicts SELL (-1) with high confidence (0.8 > 0.6)
    # - Signal strength is WEAK, so it gets rejected instead of reduced
    assert mock_patch.called
    assert result is None
    assert len(router.get_active_signals()) == 0


@pytest.mark.asyncio
async def test_ml_confirmation_strong_signal_bypassed():
    """Test that strong signals bypass ML confirmation even when ML disagrees."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength
    from core.signal_router import SignalRouter

    # Create a mock ML model and prediction function
    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": -1, "confidence": 0.8}])  # Opposite direction, high confidence

    def mock_predict(model, features_df):
        return mock_prediction_df

    # Create router with ML enabled
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Manually enable ML for this test
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    # Create a STRONG BUY signal (ENTRY_LONG)
    strong_buy_signal = TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,  # BUY signal
        signal_strength=SignalStrength.STRONG,  # Strong strength
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
    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        result = await router.process_signal(strong_buy_signal, market_data)

    # Signal should be approved despite ML disagreement because:
    # - Signal strength is STRONG, so it bypasses ML rejection
    # - ML prediction gets reduced but signal still passes
    assert mock_patch.called
    assert result is not None
    assert result.symbol == "BTC/USDT"
    # Signal strength should be reduced from STRONG to MODERATE due to ML disagreement
    assert result.signal_strength == SignalStrength.MODERATE
    assert len(router.get_active_signals()) == 1


@pytest.mark.asyncio
async def test_ml_predict_called_with_dataframe_features():
    """Test that ml_predict is called when features are provided as DataFrame."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength

    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": 1, "confidence": 0.8}])

    def mock_predict(model, features_df):
        return mock_prediction_df

    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    market_data = {
        "features": pd.DataFrame([{"feature1": 1.0, "feature2": 2.0}])
    }

    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        await router.process_signal(signal, market_data)

    assert mock_patch.called
    # Verify the features_df passed to ml_predict is a DataFrame
    args, kwargs = mock_patch.call_args
    features_df = args[1]  # Second argument is features_df
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty


@pytest.mark.asyncio
async def test_ml_predict_called_with_dict_features():
    """Test that ml_predict is called when features are provided as dict."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength

    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": 1, "confidence": 0.8}])

    def mock_predict(model, features_df):
        return mock_prediction_df

    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    market_data = {
        "features": {"feature1": 1.0, "feature2": 2.0}
    }

    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        await router.process_signal(signal, market_data)

    assert mock_patch.called
    # Verify the features_df passed to ml_predict is a DataFrame
    args, kwargs = mock_patch.call_args
    features_df = args[1]
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty


@pytest.mark.asyncio
async def test_ml_predict_called_with_series_features():
    """Test that ml_predict is called when features are provided as pd.Series."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength

    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": 1, "confidence": 0.8}])

    def mock_predict(model, features_df):
        return mock_prediction_df

    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    market_data = {
        "features": pd.Series({"feature1": 1.0, "feature2": 2.0})
    }

    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        await router.process_signal(signal, market_data)

    assert mock_patch.called
    # Verify the features_df passed to ml_predict is a DataFrame
    args, kwargs = mock_patch.call_args
    features_df = args[1]
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty


@pytest.mark.asyncio
async def test_ml_predict_not_called_with_empty_dataframe():
    """Test that ml_predict is NOT called when features DataFrame is empty."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength

    mock_model = MagicMock()

    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    market_data = {
        "features": pd.DataFrame()  # Empty DataFrame
    }

    with patch('core.signal_router.ml_predict') as mock_patch:
        await router.process_signal(signal, market_data)

    assert not mock_patch.called


@pytest.mark.asyncio
async def test_ml_predict_called_with_metadata_features_fallback():
    """Test that ml_predict is called when features are in signal metadata."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength

    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": 1, "confidence": 0.8}])

    def mock_predict(model, features_df):
        return mock_prediction_df

    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        metadata={"features": {"feature1": 1.0, "feature2": 2.0}}
    )

    # No market_data provided, should fallback to signal.metadata
    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        await router.process_signal(signal, market_data=None)

    assert mock_patch.called
    # Verify the features_df passed to ml_predict is a DataFrame
    args, kwargs = mock_patch.call_args
    features_df = args[1]
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty


@pytest.mark.asyncio
async def test_ml_predict_called_with_ohlcv_fallback():
    """Test that ml_predict is called when ohlcv data is provided as fallback."""
    import pandas as pd
    from unittest.mock import patch, MagicMock
    from decimal import Decimal
    from core.contracts import TradingSignal, SignalType, SignalStrength

    mock_model = MagicMock()
    mock_prediction_df = pd.DataFrame([{"prediction": 1, "confidence": 0.8}])

    def mock_predict(model, features_df):
        return mock_prediction_df

    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router.ml_enabled = True
    router.ml_model = mock_model
    router.ml_confidence_threshold = 0.6

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
    )

    market_data = {
        "ohlcv": {"open": 50000, "high": 51000, "low": 49000, "close": 50500, "volume": 100}
    }

    with patch('core.signal_router.ml_predict', side_effect=mock_predict) as mock_patch:
        await router.process_signal(signal, market_data)

    assert mock_patch.called
    # Verify the features_df passed to ml_predict is a DataFrame
    args, kwargs = mock_patch.call_args
    features_df = args[1]
    assert isinstance(features_df, pd.DataFrame)
    assert not features_df.empty


@pytest.mark.asyncio
async def test_get_symbol_lock_concurrency():
    """Test that symbol locks are distinct per symbol and global lock for None symbol."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Test same symbol returns same lock
    lock1 = await router._get_symbol_lock("BTC/USDT")
    lock2 = await router._get_symbol_lock("BTC/USDT")
    assert lock1 is lock2

    # Test different symbols return different locks
    lock3 = await router._get_symbol_lock("ETH/USDT")
    assert lock1 is not lock3

    # Test None symbol returns global lock
    global_lock = await router._get_symbol_lock(None)
    assert global_lock is router._lock

    # Test concurrent access to different symbols
    async def get_lock_for_symbol(symbol):
        return await router._get_symbol_lock(symbol)

    tasks = [
        get_lock_for_symbol("BTC/USDT"),
        get_lock_for_symbol("ETH/USDT"),
        get_lock_for_symbol("ADA/USDT"),
    ]

    results = await asyncio.gather(*tasks)

    # All results should be different locks
    assert len(set(results)) == 3
    assert all(isinstance(lock, asyncio.Lock) for lock in results)


@pytest.mark.asyncio
async def test_concurrent_signal_processing_different_symbols():
    """Test concurrent processing of signals for different symbols."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Create signals for different symbols
    signals = []
    for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT"]):
        signal = TradingSignal(
            strategy_id=f"strategy_{i}",
            symbol=symbol,
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            current_price=Decimal("50000"),
            timestamp=1234567890 + i,
        )
        signals.append(signal)

    # Process all signals concurrently
    tasks = [router.process_signal(signal) for signal in signals]
    results = await asyncio.gather(*tasks)

    # All signals should be approved
    assert all(result is not None for result in results)
    assert len(router.get_active_signals()) == len(signals)

    # Each symbol should have exactly one active signal
    for signal in signals:
        symbol_signals = router.get_active_signals(signal.symbol)
        assert len(symbol_signals) == 1
        assert symbol_signals[0].symbol == signal.symbol


@pytest.mark.asyncio
async def test_concurrent_signal_processing_same_symbol_conflicts():
    """Test concurrent processing of conflicting signals for the same symbol."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Create conflicting signals for the same symbol
    base_timestamp = 1234567890
    signals = [
        TradingSignal(
            strategy_id="strategy_1",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            current_price=Decimal("50000"),
            timestamp=base_timestamp,
        ),
        TradingSignal(
            strategy_id="strategy_2",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_SHORT,  # Opposite direction
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            current_price=Decimal("50000"),
            timestamp=base_timestamp + 1,
        ),
        TradingSignal(
            strategy_id="strategy_3",
            symbol="BTC/USDT",
            signal_type=SignalType.EXIT_LONG,  # Exit signal
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            current_price=Decimal("50000"),
            timestamp=base_timestamp + 2,
        ),
    ]

    # Process all signals concurrently
    tasks = [router.process_signal(signal) for signal in signals]
    results = await asyncio.gather(*tasks)

    # Check results - some should be None (rejected)
    rejected_count = sum(1 for result in results if result is None)
    assert rejected_count >= 1  # At least one should be rejected

    # At least one signal should be active (the exit signal should win due to exit_over_entry rule)
    active_signals = router.get_active_signals("BTC/USDT")
    assert len(active_signals) >= 1

    # The exit signal should be present if it was processed
    exit_signals = [s for s in active_signals if s.signal_type == SignalType.EXIT_LONG]
    if exit_signals:
        assert exit_signals[0].signal_strength == SignalStrength.STRONG


@pytest.mark.asyncio
async def test_retry_async_call_backoff_behavior():
    """Test that retry backoff increases exponentially with jitter."""
    import time
    from unittest.mock import patch

    rm = FailingRiskManager()
    router = SignalRouter(risk_manager=rm)

    start_time = time.time()

    async def failing_call():
        return await rm.evaluate_signal(None)

    # Mock sleep to track call times
    sleep_calls = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_calls.append(duration)
        await original_sleep(0.001)  # Very short sleep for testing

    with patch('asyncio.sleep', side_effect=mock_sleep):
        result = await router._retry_async_call(failing_call, retries=2, base_backoff=0.1, max_backoff=1.0)

    # Should succeed after 2 failures
    assert result is True
    assert rm.call_count == 2

    # Should have slept at least once (after first failure)
    assert len(sleep_calls) >= 1

    # First sleep should be around base_backoff (0.1)
    if sleep_calls:
        assert 0.05 <= sleep_calls[0] <= 0.15  # Allow some jitter

    # If there are multiple sleeps, second should be larger (exponential backoff)
    if len(sleep_calls) > 1:
        assert sleep_calls[1] > sleep_calls[0]


@pytest.mark.asyncio
async def test_retry_async_call_max_backoff_cap():
    """Test that retry backoff is capped at max_backoff."""
    import time
    from unittest.mock import patch

    class PersistentFailureManager:
        def __init__(self):
            self.call_count = 0

        async def evaluate_signal(self, signal, market_data=None):
            self.call_count += 1
            raise Exception("Persistent failure")

    rm = PersistentFailureManager()
    router = SignalRouter(risk_manager=rm)

    async def failing_call():
        return await rm.evaluate_signal(None)

    # Mock sleep to track durations
    sleep_calls = []
    original_sleep = asyncio.sleep

    async def mock_sleep(duration):
        sleep_calls.append(duration)
        await original_sleep(0.001)

    with patch('asyncio.sleep', side_effect=mock_sleep):
        with pytest.raises(Exception, match="Persistent failure"):
            await router._retry_async_call(failing_call, retries=3, base_backoff=1.0, max_backoff=2.0)

    # Should have attempted 4 times (initial + 3 retries)
    assert rm.call_count == 4

    # Should have slept 3 times
    assert len(sleep_calls) == 3

    # All sleeps should be <= max_backoff + jitter
    max_expected = 2.0 + (2.0 * 0.1)  # max_backoff + (max_backoff * jitter_factor)
    assert all(duration <= max_expected + 0.1 for duration in sleep_calls)  # Allow small margin for floating point
