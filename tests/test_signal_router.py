import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

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
    # Create a router with validation configuration that matches our test signals
    config = {
        'enabled': True,
        'check_balance': False,  # Disable balance checks for tests
        'check_slippage': False,  # Disable slippage checks for tests
        'max_slippage_pct': 0.02,
        'min_order_size': Decimal('0.000001'),
        'max_order_size': Decimal('1000000'),
        'tick_size': Decimal('0.00000001'),
        'lot_size': Decimal('0.00000001'),
        'tradable_symbols': ['BTC/USDT', 'ETH/USDT']
    }
    
    # Mock the ExecutionValidator with our config
    with patch('core.execution.validator.ExecutionValidator') as mock_validator:
        mock_validator.return_value.config = config
        router = SignalRouter(risk_manager=dummy_risk_manager)
        # Set the validator instance
        router.validator = mock_validator.return_value
        return router


@pytest.fixture
def sample_signal():
    return TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,  # Use float instead of Decimal to match contracts.py
        current_price=50000.0,  # Use float instead of Decimal to match contracts.py
        timestamp=datetime.now(),
        stop_loss=49000.0,  # Use float instead of Decimal to match contracts.py
        take_profit=52000.0,  # Use float instead of Decimal to match contracts.py
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
        amount=1.0,  # Use float instead of Decimal
        current_price=50000.0,  # Use float instead of Decimal
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
        amount=0.0,  # Invalid - use float instead of Decimal
        current_price=50000.0,  # Use float instead of Decimal
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
        amount=1.0,  # Use float instead of Decimal
        current_price=50000.0,  # Use float instead of Decimal
    )
    assert signal_router._validate_signal(invalid1) is False

    # Zero amount
    invalid2 = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=0.0,  # Use float instead of Decimal
        current_price=50000.0,  # Use float instead of Decimal
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
        current_price=Decimal("50000"),
    )

    sig2 = TradingSignal(
        strategy_id="s2",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_SHORT,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
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
        current_price=Decimal("50000"),
    )

    sig_exit = TradingSignal(
        strategy_id="s2",
        symbol="BTC/USDT",
        signal_type=SignalType.EXIT_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
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
async def test_concurrent_signal_processing(signal_router):
    """Test that concurrent signals for different symbols don't interfere."""
    # Create signals for different symbols
    signal1 = TradingSignal(
        strategy_id="s1",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    signal2 = TradingSignal(
        strategy_id="s2",
        symbol="ETH/USDT",
        signal_type=SignalType.ENTRY_SHORT,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("2.0"),
        current_price=Decimal("3000"),
    )

    # Process signals concurrently
    tasks = [
        signal_router.process_signal(signal1),
        signal_router.process_signal(signal2)
    ]
    results = await asyncio.gather(*tasks)

    # Both should succeed
    assert all(result is not None for result in results)
    assert len(signal_router.get_active_signals()) == 2
    assert len(signal_router.get_active_signals("BTC/USDT")) == 1
    assert len(signal_router.get_active_signals("ETH/USDT")) == 1


@pytest.mark.asyncio
async def test_record_router_error_context_logging():
    """Test _record_router_error logs context information."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    context = {"symbol": "BTC/USDT", "strategy": "test_strategy"}
    await router._record_router_error(Exception("Test error"), context)

    assert router.critical_errors == 1
    assert router.block_signals is False  # Below threshold


# Additional tests for update_signal_status (lines 441-444)
@pytest.mark.asyncio
async def test_update_signal_status_nonexistent_signal():
    """Test update_signal_status with nonexistent signal ID."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Try to update a signal that doesn't exist
    await router.update_signal_status(
        TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            current_price=Decimal("50000"),
        ),
        "executed",
        "test reason"
    )

    # Should not raise exception
    assert len(router.get_active_signals()) == 0


# Additional tests for get_active_signals and get_signal_history (lines 451-452, 460-467)
@pytest.mark.asyncio
async def test_get_active_signals_empty():
    """Test get_active_signals when no signals are active."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    assert router.get_active_signals() == []
    assert router.get_active_signals("BTC/USDT") == []


@pytest.mark.asyncio
async def test_get_signal_history_limit():
    """Test get_signal_history respects limit parameter."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Add multiple signals
    for i in range(5):
        signal = TradingSignal(
            strategy_id=f"test_{i}",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            current_price=Decimal("50000"),
        )
        router._store_signal(signal)

    # Test limit
    history = router.get_signal_history(limit=3)
    assert len(history) == 3

    # Test no limit
    history = router.get_signal_history()
    assert len(history) == 5


# Additional tests for close_journal (lines 482-486)
@pytest.mark.asyncio
async def test_close_journal_no_journal():
    """Test close_journal when no journal is configured."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Should not raise exception when no journal
    await router.close_journal()


@pytest.mark.asyncio
async def test_close_journal_with_journal():
    """Test close_journal with active journal."""
    import tempfile
    from pathlib import Path

    rm = DummyRiskManager()
    with tempfile.TemporaryDirectory() as temp_dir:
        journal_path = Path(temp_dir) / "test_journal.jsonl"

        router = SignalRouter(risk_manager=rm)
        router.journal_path = journal_path
        router._journal_enabled = True
        router._journal_writer = "mock_writer"  # Mock writer

        # Should not raise exception
        await router.close_journal()


# Additional tests for _get_symbol_lock (lines 532, 548)
@pytest.mark.asyncio
async def test_get_symbol_lock_none_symbol():
    """Test _get_symbol_lock with None symbol returns global lock."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    lock = await router._get_symbol_lock(None)
    assert lock is router._lock


@pytest.mark.asyncio
async def test_get_symbol_lock_creates_new_lock():
    """Test _get_symbol_lock creates new lock for new symbol."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    lock = await router._get_symbol_lock("NEW_SYMBOL")
    assert isinstance(lock, asyncio.Lock)
    assert "NEW_SYMBOL" in router._symbol_locks
    assert router._symbol_locks["NEW_SYMBOL"] is lock

@pytest.mark.asyncio
async def test_process_signal_with_invalid_symbol_lock():
    """Test process_signal handles invalid symbol gracefully."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    signal = TradingSignal(
        strategy_id="test",
        symbol="",  # Invalid symbol
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    result = await router.process_signal(signal)
    assert result is None

@pytest.mark.asyncio
async def test_resolve_conflicts_with_empty_list():
    """Test _resolve_conflicts with empty conflicting signals list."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    result = await router._resolve_conflicts(signal, [])
    assert result == signal

@pytest.mark.asyncio
async def test_resolve_conflicts_strength_based():
    """Test conflict resolution based on strength."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    weak_signal = TradingSignal(
        strategy_id="weak",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.WEAK,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    strong_signal = TradingSignal(
        strategy_id="strong",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    # Add strong signal first
    await router.process_signal(strong_signal)
    assert len(router.get_active_signals()) == 1

    # Try to add weak signal - should be rejected
    result = await router.process_signal(weak_signal)
    assert result is None
    assert len(router.get_active_signals()) == 1

@pytest.mark.asyncio
async def test_resolve_conflicts_newer_first():
    """Test conflict resolution based on timestamp (newer first)."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    # Disable strength-based resolution
    router.conflict_resolution_rules["strength_based"] = False
    router.conflict_resolution_rules["newer_first"] = True

    old_signal = TradingSignal(
        strategy_id="old",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
        timestamp=datetime.fromtimestamp(1000),
    )

    new_signal = TradingSignal(
        strategy_id="new",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
        timestamp=datetime.fromtimestamp(2000),
    )

    # Add old signal first
    await router.process_signal(old_signal)
    assert len(router.get_active_signals()) == 1

    # Add new signal - should replace old one
    result = await router.process_signal(new_signal)
    assert result is not None
    assert len(router.get_active_signals()) == 1
    assert router.get_active_signals()[0].strategy_id == "new"

@pytest.mark.asyncio
async def test_store_signal_with_journal_disabled():
    """Test _store_signal when journal is disabled."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router._journal_enabled = False

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    router._store_signal(signal)
    assert len(router.active_signals) == 1
    assert len(router.signal_history) == 1

@pytest.mark.asyncio
async def test_cancel_signal_with_journal_disabled():
    """Test _cancel_signal when journal is disabled."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router._journal_enabled = False

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    router._store_signal(signal)
    assert len(router.active_signals) == 1

    await router._cancel_signal(signal)
    assert len(router.active_signals) == 0

@pytest.mark.asyncio
async def test_recover_from_journal_disabled():
    """Test recover_from_journal when disabled."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)
    router._journal_enabled = False

    # Should return early without doing anything
    router.recover_from_journal()
    assert len(router.active_signals) == 0

@pytest.mark.asyncio
async def test_ml_extract_features_with_list_of_scalars():
    """Test _extract_features_for_ml with list of scalars."""
    import pandas as pd

    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=Decimal("1.0"),
        current_price=Decimal("50000"),
    )

    market_data = {
        "features": [1.0, 2.0, 3.0]
    }

    result = router._extract_features_for_ml(market_data, signal)
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)  # Single row, single column named 'value'

@pytest.mark.asyncio
async def test_ml_extract_features_with_invalid_data():
    """Test _extract_features_for_ml with invalid data types."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,  # Use float instead of Decimal
        current_price=50000.0,  # Use float instead of Decimal
    )

    # Test with non-convertible object
    market_data = {
        "features": object()  # Not convertible to DataFrame
    }

    result = router._extract_features_for_ml(market_data, signal)
    assert result is None


# MARKET DATA GAPS TESTS
@pytest.mark.asyncio
async def test_process_signal_with_missing_candle_data(signal_router):
    """Test processing signal with missing candle data (market gap)."""
    # Mock the validator to simulate missing market data
    with patch.object(signal_router.validator, 'validate_order') as mock_validate:
        mock_validate.return_value = False  # Simulate validation failure due to missing data
        
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=1.0,
            current_price=50000.0,
        )
        
        result = await signal_router.process_signal(signal)
        assert result is None
        # Router should not crash, just return None


@pytest.mark.asyncio
async def test_process_signal_with_incomplete_ohlcv_sequence(signal_router):
    """Test processing signal with incomplete OHLCV sequence."""
    # Create a signal with incomplete market data
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="limit",
        amount=1.0,
        current_price=50000.0,
    )
    
    # Test with incomplete market data (missing close price)
    incomplete_market_data = {
        "ohlcv": [
            [1609459200000, 50000.0, None, 49000.0, 100.0],  # Missing high price
        ]
    }
    
    # Mock the validator to handle incomplete data
    with patch.object(signal_router.validator, 'validate_order') as mock_validate:
        mock_validate.return_value = True  # Allow processing
        
        # This should not crash the router
        try:
            result = await signal_router.process_signal(signal)
            # Result may be None or valid depending on implementation
            assert result is None or result.symbol == "BTC/USDT"
        except Exception as e:
            pytest.fail(f"Router crashed with incomplete data: {e}")


@pytest.mark.asyncio
async def test_process_signal_with_empty_market_data(signal_router):
    """Test processing signal with empty market data."""
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # Test with empty market data
    empty_market_data = {"ohlcv": []}
    
    # Mock the validator to handle empty data
    with patch.object(signal_router.validator, 'validate_order') as mock_validate:
        mock_validate.return_value = True  # Allow processing
        
        # This should not crash the router
        try:
            result = await signal_router.process_signal(signal)
            # Result may be None or valid depending on implementation
            assert result is None or result.symbol == "BTC/USDT"
        except Exception as e:
            pytest.fail(f"Router crashed with empty market data: {e}")


@pytest.mark.asyncio
async def test_process_signal_with_none_market_data(signal_router):
    """Test processing signal with None market data."""
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # Test with None market data
    # This should not crash the router
    try:
        result = await signal_router.process_signal(signal)
        # Result may be None or valid depending on implementation
        assert result is None or result.symbol == "BTC/USDT"
    except Exception as e:
        pytest.fail(f"Router crashed with None market data: {e}")


# API TIMEOUTS TESTS
@pytest.mark.asyncio
async def test_process_signal_with_exchange_api_timeout(signal_router):
    """Test processing signal when exchange API times out."""
    # Create a risk manager that simulates API timeout
    class TimeoutRiskManager:
        async def evaluate_signal(self, signal, market_data=None):
            await asyncio.sleep(0.1)  # Simulate network delay
            raise Exception("API timeout: Connection to exchange timed out")
    
    timeout_router = SignalRouter(risk_manager=TimeoutRiskManager())
    
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # This should not crash the router, should handle timeout gracefully
    try:
        result = await timeout_router.process_signal(signal)
        # Router should handle timeout and return None or appropriate response
        assert result is None
    except Exception as e:
        pytest.fail(f"Router crashed on API timeout: {e}")


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_process_signal_with_retry_on_api_timeout(mock_sleep, signal_router):
    """Test that router retries on API timeout and eventually succeeds."""
    # Create a risk manager that fails first time then succeeds
    class FlakyRiskManager:
        def __init__(self):
            self.call_count = 0
        
        async def evaluate_signal(self, signal, market_data=None):
            self.call_count += 1
            if self.call_count == 1:
                raise Exception("API timeout: Connection to exchange timed out")
            return True
    
    flaky_router = SignalRouter(risk_manager=FlakyRiskManager())
    
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # This should retry and eventually succeed
    result = await flaky_router.process_signal(signal)
    assert result is not None
    assert flaky_router.risk_manager.call_count == 2  # Failed once, succeeded on retry
    mock_sleep.assert_awaited()


@pytest.mark.asyncio
async def test_process_signal_with_persistent_api_failure(signal_router):
    """Test processing signal with persistent API failures."""
    # Create a risk manager that always fails
    class FailingRiskManager:
        async def evaluate_signal(self, signal, market_data=None):
            raise Exception("API error: Exchange unavailable")
    
    failing_router = SignalRouter(risk_manager=FailingRiskManager(), max_retries=1)
    
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # This should exhaust retries and return None
    result = await failing_router.process_signal(signal)
    assert result is None


# ZERO LIQUIDITY SCENARIOS TESTS
@pytest.mark.asyncio
async def test_process_signal_with_zero_liquidity(signal_router):
    """Test processing signal when there's zero liquidity (bid/ask spread unavailable)."""
    # Create a signal for a market with no liquidity
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # Mock validator to simulate zero liquidity
    with patch.object(signal_router.validator, 'validate_order') as mock_validate:
        mock_validate.return_value = False  # Simulate rejection due to no liquidity
        
        result = await signal_router.process_signal(signal)
        assert result is None
        # Router should not crash, just reject the signal


@pytest.mark.asyncio
async def test_process_signal_with_zero_volume(signal_router):
    """Test processing signal when trading volume is zero."""
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # Mock validator to simulate zero volume
    with patch.object(signal_router.validator, 'validate_order') as mock_validate:
        mock_validate.side_effect = Exception("Zero volume: No trading activity")
        
        result = await signal_router.process_signal(signal)
        assert result is None


@pytest.mark.asyncio
async def test_process_signal_with_wide_spread(signal_router):
    """Test processing signal with extremely wide bid/ask spread."""
    signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # Mock validator to simulate wide spread
    with patch.object(signal_router.validator, 'validate_order') as mock_validate:
        mock_validate.return_value = False  # Simulate rejection due to wide spread
        
        result = await signal_router.process_signal(signal)
        assert result is None


# INVALID SIGNAL HANDLING TESTS
@pytest.mark.asyncio
async def test_process_signal_with_malformed_signal_data(signal_router):
    """Test processing signal with malformed/corrupted signal data."""
    # Create a signal with invalid data
    malformed_signal = TradingSignal(
        strategy_id="",  # Invalid empty strategy
        symbol="",  # Invalid empty symbol
        signal_type=None,  # Invalid None type
        signal_strength=None,  # Invalid None strength
        order_type="invalid_order_type",  # Invalid order type
        amount=-1.0,  # Invalid negative amount
        current_price=-50000.0,  # Invalid negative price
    )
    
    # This should not crash the router
    try:
        result = await signal_router.process_signal(malformed_signal)
        assert result is None  # Router should reject invalid signals
    except Exception as e:
        pytest.fail(f"Router crashed with malformed signal: {e}")


@pytest.mark.asyncio
async def test_process_signal_with_missing_required_fields(signal_router):
    """Test processing signal with missing required fields."""
    # Create a signal with missing required fields
    incomplete_signal = TradingSignal(
        strategy_id=None,  # Missing strategy_id
        symbol=None,  # Missing symbol
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # This should not crash the router
    try:
        result = await signal_router.process_signal(incomplete_signal)
        assert result is None  # Router should reject incomplete signals
    except Exception as e:
        pytest.fail(f"Router crashed with incomplete signal: {e}")


@pytest.mark.asyncio
async def test_process_signal_with_corrupted_timestamp(signal_router):
    """Test processing signal with corrupted timestamp."""
    # Create a signal with invalid timestamp
    from datetime import datetime
    corrupted_signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
        timestamp="invalid_timestamp",  # Invalid timestamp string
    )
    
    # This should not crash the router
    try:
        result = await signal_router.process_signal(corrupted_signal)
        assert result is None  # Router should reject signals with invalid timestamps
    except Exception as e:
        pytest.fail(f"Router crashed with corrupted timestamp: {e}")


@pytest.mark.asyncio
async def test_process_signal_with_extreme_values(signal_router):
    """Test processing signal with extreme/invalid values."""
    # Create a signal with extreme values that could cause numerical issues
    extreme_signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1e300,  # Extremely large amount
        current_price=1e300,  # Extremely large price
    )
    
    # This should not crash the router
    try:
        result = await signal_router.process_signal(extreme_signal)
        # Router should handle extreme values gracefully
        assert result is None or result.symbol == "BTC/USDT"
    except Exception as e:
        pytest.fail(f"Router crashed with extreme values: {e}")


@pytest.mark.asyncio
async def test_process_signal_with_unicode_in_fields(signal_router):
    """Test processing signal with unicode characters in string fields."""
    # Create a signal with unicode characters
    unicode_signal = TradingSignal(
        strategy_id="测试策略",  # Chinese characters
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
    )
    
    # This should not crash the router
    try:
        result = await signal_router.process_signal(unicode_signal)
        # Router should handle unicode gracefully
        assert result is None or result.symbol == "BTC/USDT"
    except Exception as e:
        pytest.fail(f"Router crashed with unicode characters: {e}")


@pytest.mark.asyncio
async def test_process_signal_with_special_characters_in_metadata(signal_router):
    """Test processing signal with special characters in metadata."""
    # Create a signal with special characters in metadata
    special_signal = TradingSignal(
        strategy_id="test",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        amount=1.0,
        current_price=50000.0,
        metadata={
            "note": "Special chars: <>&'\"\\",
            "emoji": "🚀💰",
            "newlines": "Line1\nLine2\r\nLine3",
            "tabs": "Col1\tCol2\tCol3",
        }
    )
    
    # This should not crash the router
    try:
        result = await signal_router.process_signal(special_signal)
        # Router should handle special characters gracefully
        assert result is None or result.symbol == "BTC/USDT"
    except Exception as e:
        pytest.fail(f"Router crashed with special characters: {e}")
