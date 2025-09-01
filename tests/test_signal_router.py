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
async def test_retry_async_call_success():
    """Test successful retry of async call."""
    rm = DummyRiskManager()
    router = SignalRouter(risk_manager=rm)

    async def dummy_call():
        return "success"

    result = await router._retry_async_call(dummy_call, retries=2)
    assert result == "success"


@pytest.mark.asyncio
async def test_retry_async_call_with_failure():
    """Test retry mechanism with temporary failure."""
    rm = FailingRiskManager()
    router = SignalRouter(risk_manager=rm)

    async def failing_call():
        return await rm.evaluate_signal(None)

    result = await router._retry_async_call(failing_call, retries=2)
    assert result is True
    assert rm.call_count == 2  # Failed once, succeeded on retry


@pytest.mark.asyncio
async def test_retry_async_call_exhaust_retries():
    """Test retry exhaustion."""
    rm = FailingRiskManager()
    router = SignalRouter(risk_manager=rm)

    async def always_fail():
        raise Exception("Always fails")

    with pytest.raises(Exception, match="Always fails"):
        await router._retry_async_call(always_fail, retries=1)


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
