import pytest
from decimal import Decimal
from unittest.mock import MagicMock
import dataclasses
from enum import Enum
from datetime import datetime

from utils.adapter import signal_to_dict, _normalize_timestamp
from core.contracts import TradingSignal, SignalType, SignalStrength


class MockOrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class MockSignalWithToDict:
    """Mock signal object with to_dict method."""

    def __init__(self):
        self.id = "test_id"
        self.symbol = "BTC/USDT"
        self.amount = Decimal("1.0")
        self.price = Decimal("50000")

    def to_dict(self):
        return {
            "id": self.id,
            "symbol": self.symbol,
            "amount": str(self.amount),
            "price": str(self.price),
        }


class MockSignalWithAttributes:
    """Mock signal object with common attributes."""

    def __init__(self):
        self.id = "test_id"
        self.symbol = "BTC/USDT"
        self.order_type = MockOrderType.MARKET
        self.amount = Decimal("1.0")
        self.price = Decimal("50000")
        self.stop_loss = Decimal("49000")
        self.take_profit = Decimal("52000")
        self.timestamp = 1234567890
        self._private_attr = "should_not_appear"


class MockSignalWithBrokenToDict:
    """Mock signal with broken to_dict method."""

    def to_dict(self):
        raise ValueError("Broken to_dict")


class MockSignalWithBrokenProperty:
    """Mock signal with broken property."""

    @property
    def symbol(self):
        raise AttributeError("Broken property")


@pytest.fixture
def sample_trading_signal():
    """Create a sample TradingSignal dataclass."""
    return TradingSignal(
        strategy_id="test_strategy",
        symbol="BTC/USDT",
        signal_type=SignalType.ENTRY_LONG,
        signal_strength=SignalStrength.STRONG,
        order_type="market",
        quantity=Decimal("1.0"),
        side="buy",
        price=Decimal("50000"),
        current_price=Decimal("50000"),
        timestamp=1234567890,
        stop_loss=Decimal("49000"),
        take_profit=Decimal("52000"),
    )


def test_signal_to_dict_none():
    """Test signal_to_dict with None input."""
    result = signal_to_dict(None)
    assert result == {}


def test_signal_to_dict_dict():
    """Test signal_to_dict with dict input."""
    input_dict = {
        "symbol": "BTC/USDT",
        "amount": Decimal("1.0"),
        "price": Decimal("50000"),
    }
    result = signal_to_dict(input_dict)
    assert result == input_dict
    assert result is not input_dict  # Should be a copy


def test_signal_to_dict_dataclass(sample_trading_signal):
    """Test signal_to_dict with dataclass input."""
    result = signal_to_dict(sample_trading_signal)

    # Check that all dataclass fields are present
    assert result["strategy_id"] == "test_strategy"
    assert result["symbol"] == "BTC/USDT"
    assert result["signal_type"] == 1  # SignalType.ENTRY_LONG.value
    assert result["signal_strength"] == 3  # SignalStrength.STRONG.value
    assert result["order_type"] == "market"
    assert result["amount"] == Decimal("1.0")
    assert result["price"] == Decimal("50000")
    assert result["current_price"] == Decimal("50000")
    # Note: timestamp might be converted to milliseconds in TradingSignal
    assert result["timestamp"] == 1234567890000  # May be converted to ms
    assert result["stop_loss"] == Decimal("49000")
    assert result["take_profit"] == Decimal("52000")


def test_signal_to_dict_with_to_dict():
    """Test signal_to_dict with object that has to_dict method."""
    signal = MockSignalWithToDict()
    result = signal_to_dict(signal)

    expected = {
        "id": "test_id",
        "symbol": "BTC/USDT",
        "amount": "1.0",  # Note: to_dict returns strings
        "price": "50000",
    }
    assert result == expected


def test_signal_to_dict_with_attributes():
    """Test signal_to_dict with object that has common attributes."""
    signal = MockSignalWithAttributes()
    result = signal_to_dict(signal)

    # Check common attributes are extracted
    assert result["id"] == "test_id"
    assert result["symbol"] == "BTC/USDT"
    assert result["order_type"] == "market"  # Enum converted to value
    assert result["amount"] == Decimal("1.0")
    assert result["price"] == Decimal("50000")
    assert result["stop_loss"] == Decimal("49000")
    assert result["take_profit"] == Decimal("52000")
    assert result["timestamp"] == 1234567890

    # Private attributes should not be included
    assert "_private_attr" not in result


def test_signal_to_dict_with_enum_conversion():
    """Test that enums are properly converted."""
    signal = MockSignalWithAttributes()
    result = signal_to_dict(signal)

    # order_type should be converted from enum to string value
    assert result["order_type"] == "market"
    assert isinstance(result["order_type"], str)


def test_signal_to_dict_with_broken_to_dict():
    """Test signal_to_dict with broken to_dict method falls back to attribute probing."""
    signal = MockSignalWithBrokenToDict()
    # Add some attributes for fallback
    signal.symbol = "BTC/USDT"
    signal.amount = Decimal("1.0")

    result = signal_to_dict(signal)

    # Should fall back to attribute probing
    assert result["symbol"] == "BTC/USDT"
    assert result["amount"] == Decimal("1.0")


def test_signal_to_dict_with_broken_property():
    """Test signal_to_dict handles broken properties gracefully."""
    signal = MockSignalWithBrokenProperty()
    signal.amount = Decimal("1.0")  # Add a working attribute

    result = signal_to_dict(signal)

    # Broken property should be skipped, working ones included
    assert "symbol" not in result  # Broken property skipped
    assert result["amount"] == Decimal("1.0")


def test_signal_to_dict_empty_object():
    """Test signal_to_dict with object that has no useful attributes."""
    class EmptyObject:
        pass

    signal = EmptyObject()
    result = signal_to_dict(signal)

    assert result == {}


def test_signal_to_dict_with_callable_attributes():
    """Test that callable attributes are included in __dict__ fallback (current behavior)."""
    class ObjectWithCallable:
        def __init__(self):
            self.symbol = "BTC/USDT"
            self.amount = Decimal("1.0")
            self.method = lambda: "callable"

    signal = ObjectWithCallable()
    result = signal_to_dict(signal)

    assert result["symbol"] == "BTC/USDT"
    assert result["amount"] == Decimal("1.0")
    # Note: Current implementation includes callable attributes from __dict__
    assert "method" in result


def test_signal_to_dict_with_private_attributes():
    """Test that private attributes (starting with _) are not included."""
    class ObjectWithPrivate:
        def __init__(self):
            self.symbol = "BTC/USDT"
            self._private = "should_not_appear"
            self.__dunder = "should_not_appear"

    signal = ObjectWithPrivate()
    result = signal_to_dict(signal)

    assert result["symbol"] == "BTC/USDT"
    assert "_private" not in result
    assert "_ObjectWithPrivate__dunder" not in result


def test_signal_to_dict_with_none_values():
    """Test signal_to_dict handles None values properly."""
    class ObjectWithNone:
        def __init__(self):
            self.symbol = "BTC/USDT"
            self.amount = None
            self.price = Decimal("50000")

    signal = ObjectWithNone()
    result = signal_to_dict(signal)

    assert result["symbol"] == "BTC/USDT"
    assert result["amount"] is None
    assert result["price"] == Decimal("50000")


def test_signal_to_dict_precedence():
    """Test that to_dict() takes precedence over attribute probing."""
    class SignalWithBoth:
        def __init__(self):
            self.symbol = "FROM_ATTR"
            self.amount = Decimal("1.0")

        def to_dict(self):
            return {
                "symbol": "FROM_TO_DICT",
                "amount": Decimal("2.0"),
            }

    signal = SignalWithBoth()
    result = signal_to_dict(signal)

    # to_dict() should take precedence
    assert result["symbol"] == "FROM_TO_DICT"
    assert result["amount"] == Decimal("2.0")


def test_signal_to_dict_dataclass_precedence():
    """Test that dataclass conversion takes precedence over to_dict()."""
    @dataclasses.dataclass
    class DataclassWithToDict:
        symbol: str
        amount: Decimal

        def to_dict(self):
            return {"symbol": "FROM_TO_DICT", "amount": Decimal("999")}

    signal = DataclassWithToDict(symbol="FROM_DATACLASS", amount=Decimal("1.0"))
    result = signal_to_dict(signal)

    # dataclass should take precedence over to_dict()
    assert result["symbol"] == "FROM_DATACLASS"
    assert result["amount"] == Decimal("1.0")


def test_signal_to_dict_with_complex_enum():
    """Test enum conversion with value attribute."""
    class ComplexEnum(Enum):
        BUY = 1
        SELL = 2

    class SignalWithComplexEnum:
        def __init__(self):
            self.side = ComplexEnum.BUY

    signal = SignalWithComplexEnum()
    result = signal_to_dict(signal)

    # Should convert to value (1) since .value is checked first
    assert result["side"] == 1


def test_signal_to_dict_with_non_enum_value_attr():
    """Test handling of objects with 'value' attribute that are not enums."""
    class NotAnEnum:
        def __init__(self):
            self.value = "not_an_enum"

    class SignalWithValue:
        def __init__(self):
            self.param = NotAnEnum()

    signal = SignalWithValue()
    result = signal_to_dict(signal)

    # Should preserve the object as-is (not try to convert it)
    assert isinstance(result["param"], NotAnEnum)
    assert result["param"].value == "not_an_enum"


def test_normalize_timestamp_int():
    """Test _normalize_timestamp with int input."""
    result = _normalize_timestamp(1234567890)
    assert result == 1234567890
    assert isinstance(result, int)


def test_normalize_timestamp_float():
    """Test _normalize_timestamp with float input (seconds since epoch)."""
    # Test with a float representing seconds
    timestamp_float = 1633017600.123
    result = _normalize_timestamp(timestamp_float)
    expected = int(1633017600.123 * 1000)  # Convert to milliseconds
    assert result == expected
    assert isinstance(result, int)


def test_normalize_timestamp_datetime():
    """Test _normalize_timestamp with datetime input."""
    from datetime import timezone
    dt = datetime(2021, 10, 1, 12, 0, 0, tzinfo=timezone.utc)
    result = _normalize_timestamp(dt)
    expected = int(dt.timestamp() * 1000)
    assert result == expected
    assert isinstance(result, int)


def test_normalize_timestamp_str():
    """Test _normalize_timestamp with ISO string input."""
    iso_str = "2021-10-01T12:00:00Z"
    dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
    expected = int(dt.timestamp() * 1000)
    result = _normalize_timestamp(iso_str)
    assert result == expected
    assert isinstance(result, int)


def test_normalize_timestamp_unsupported():
    """Test _normalize_timestamp with unsupported type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported timestamp type"):
        _normalize_timestamp([])  # List is not supported


def test_signal_to_dict_with_float_timestamp():
    """Test signal_to_dict normalizes float timestamp in attribute probe."""
    class SignalWithFloatTimestamp:
        def __init__(self):
            self.symbol = "BTC/USDT"
            self.timestamp = 1633017600.123  # Float seconds

    signal = SignalWithFloatTimestamp()
    result = signal_to_dict(signal)

    # Float should be converted to int milliseconds
    expected_ms = int(1633017600.123 * 1000)
    assert result["timestamp"] == expected_ms
    assert isinstance(result["timestamp"], int)
