from decimal import Decimal

import pytest

from core.contracts import SignalStrength, SignalType, TradingSignal


def make_signal(amount, metadata=None):
    return TradingSignal(
        strategy_id="strategy_x",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.MODERATE,
        order_type="market",
        amount=amount,
        metadata=metadata or {},
    )


def test_fraction_to_notional():
    sig = make_signal(0.1, metadata={"amount_is_fraction": True})
    sig.normalize_amount(Decimal("1000"))
    assert sig.amount == Decimal("100.00000000")


def test_no_metadata_no_change():
    sig = make_signal(Decimal("1.5"))
    before = sig.amount
    sig.normalize_amount(None)
    assert sig.amount == before


def test_missing_total_balance_raises():
    sig = make_signal(0.2, metadata={"amount_is_fraction": True})
    with pytest.raises(ValueError):
        sig.normalize_amount(None)


def test_clamp_fraction_above_one():
    sig = make_signal(2, metadata={"amount_is_fraction": True})
    sig.normalize_amount(Decimal("100"))
    assert sig.amount == Decimal("100.00000000")


def test_string_amount_conversion():
    sig = make_signal("0.25", metadata={"amount_is_fraction": True})
    sig.normalize_amount(Decimal("400"))
    assert sig.amount == Decimal("100.00000000")


def test_negative_fraction_clamped():
    sig = make_signal(-0.5, metadata={"amount_is_fraction": True})
    sig.normalize_amount(Decimal("1000"))
    assert sig.amount == Decimal("0.00000000")
