import asyncio
from decimal import Decimal

import pytest

from core.signal_router import SignalRouter
from core.contracts import TradingSignal, SignalType, SignalStrength


class DummyRiskManager:
    """Simple risk manager stub that approves all signals."""

    require_stop_loss = False

    async def evaluate_signal(self, signal, market_data=None):
        await asyncio.sleep(0)  # yield control to allow concurrency in tests
        return True


@pytest.mark.asyncio
async def test_concurrent_signal_processing_no_conflicts():
    """
    Spawn many concurrent signals for distinct symbols and ensure all are accepted
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
