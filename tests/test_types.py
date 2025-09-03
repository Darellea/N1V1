"""
tests/test_types.py

Comprehensive tests for core type definitions and enum values.
Tests validate TradingMode, OrderType, OrderStatus enums and Order dataclass.
"""

import pytest
from decimal import Decimal
from core.types import TradingMode, OrderType, OrderStatus, Order


class TestTradingMode:
    """Test cases for TradingMode enum."""

    def test_trading_mode_values(self):
        """Test TradingMode enum has expected values."""
        assert TradingMode.LIVE.value == "live"
        assert TradingMode.PAPER.value == "paper"
        assert TradingMode.BACKTEST.value == "backtest"

    def test_trading_mode_names(self):
        """Test TradingMode enum names."""
        assert TradingMode.LIVE.name == "LIVE"
        assert TradingMode.PAPER.name == "PAPER"
        assert TradingMode.BACKTEST.name == "BACKTEST"

    def test_trading_mode_iteration(self):
        """Test TradingMode enum can be iterated."""
        modes = list(TradingMode)
        assert len(modes) == 3
        assert TradingMode.LIVE in modes
        assert TradingMode.PAPER in modes
        assert TradingMode.BACKTEST in modes

    def test_trading_mode_membership(self):
        """Test TradingMode enum membership."""
        assert "live" in [mode.value for mode in TradingMode]
        assert "paper" in [mode.value for mode in TradingMode]
        assert "backtest" in [mode.value for mode in TradingMode]

    def test_trading_mode_from_string(self):
        """Test creating TradingMode from string values."""
        assert TradingMode("live") == TradingMode.LIVE
        assert TradingMode("paper") == TradingMode.PAPER
        assert TradingMode("backtest") == TradingMode.BACKTEST

    def test_trading_mode_invalid_value(self):
        """Test TradingMode with invalid value raises ValueError."""
        with pytest.raises(ValueError):
            TradingMode("invalid")

    def test_trading_mode_string_representation(self):
        """Test TradingMode string representations."""
        assert str(TradingMode.LIVE) == "TradingMode.LIVE"
        assert repr(TradingMode.LIVE) == "<TradingMode.LIVE: 'live'>"


class TestOrderType:
    """Test cases for OrderType enum."""

    def test_order_type_values(self):
        """Test OrderType enum has expected values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_LOSS.value == "stop_loss"
        assert OrderType.TAKE_PROFIT.value == "take_profit"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"

    def test_order_type_names(self):
        """Test OrderType enum names."""
        assert OrderType.MARKET.name == "MARKET"
        assert OrderType.LIMIT.name == "LIMIT"
        assert OrderType.STOP_LOSS.name == "STOP_LOSS"
        assert OrderType.TAKE_PROFIT.name == "TAKE_PROFIT"
        assert OrderType.TRAILING_STOP.name == "TRAILING_STOP"

    def test_order_type_iteration(self):
        """Test OrderType enum can be iterated."""
        order_types = list(OrderType)
        assert len(order_types) == 5
        assert OrderType.MARKET in order_types
        assert OrderType.LIMIT in order_types
        assert OrderType.STOP_LOSS in order_types
        assert OrderType.TAKE_PROFIT in order_types
        assert OrderType.TRAILING_STOP in order_types

    def test_order_type_membership(self):
        """Test OrderType enum membership."""
        values = [order_type.value for order_type in OrderType]
        assert "market" in values
        assert "limit" in values
        assert "stop_loss" in values
        assert "take_profit" in values
        assert "trailing_stop" in values

    def test_order_type_from_string(self):
        """Test creating OrderType from string values."""
        assert OrderType("market") == OrderType.MARKET
        assert OrderType("limit") == OrderType.LIMIT
        assert OrderType("stop_loss") == OrderType.STOP_LOSS
        assert OrderType("take_profit") == OrderType.TAKE_PROFIT
        assert OrderType("trailing_stop") == OrderType.TRAILING_STOP

    def test_order_type_invalid_value(self):
        """Test OrderType with invalid value raises ValueError."""
        with pytest.raises(ValueError):
            OrderType("invalid_order_type")

    def test_order_type_string_representation(self):
        """Test OrderType string representations."""
        assert str(OrderType.MARKET) == "OrderType.MARKET"
        assert repr(OrderType.MARKET) == "<OrderType.MARKET: 'market'>"


class TestOrderStatus:
    """Test cases for OrderStatus enum."""

    def test_order_status_values(self):
        """Test OrderStatus enum has expected values."""
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.CANCELED.value == "canceled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"

    def test_order_status_names(self):
        """Test OrderStatus enum names."""
        assert OrderStatus.OPEN.name == "OPEN"
        assert OrderStatus.FILLED.name == "FILLED"
        assert OrderStatus.PARTIALLY_FILLED.name == "PARTIALLY_FILLED"
        assert OrderStatus.CANCELED.name == "CANCELED"
        assert OrderStatus.REJECTED.name == "REJECTED"
        assert OrderStatus.EXPIRED.name == "EXPIRED"

    def test_order_status_iteration(self):
        """Test OrderStatus enum can be iterated."""
        statuses = list(OrderStatus)
        assert len(statuses) == 6
        assert OrderStatus.OPEN in statuses
        assert OrderStatus.FILLED in statuses
        assert OrderStatus.PARTIALLY_FILLED in statuses
        assert OrderStatus.CANCELED in statuses
        assert OrderStatus.REJECTED in statuses
        assert OrderStatus.EXPIRED in statuses

    def test_order_status_membership(self):
        """Test OrderStatus enum membership."""
        values = [status.value for status in OrderStatus]
        assert "open" in values
        assert "filled" in values
        assert "partially_filled" in values
        assert "canceled" in values
        assert "rejected" in values
        assert "expired" in values

    def test_order_status_from_string(self):
        """Test creating OrderStatus from string values."""
        assert OrderStatus("open") == OrderStatus.OPEN
        assert OrderStatus("filled") == OrderStatus.FILLED
        assert OrderStatus("partially_filled") == OrderStatus.PARTIALLY_FILLED
        assert OrderStatus("canceled") == OrderStatus.CANCELED
        assert OrderStatus("rejected") == OrderStatus.REJECTED
        assert OrderStatus("expired") == OrderStatus.EXPIRED

    def test_order_status_invalid_value(self):
        """Test OrderStatus with invalid value raises ValueError."""
        with pytest.raises(ValueError):
            OrderStatus("invalid_status")

    def test_order_status_string_representation(self):
        """Test OrderStatus string representations."""
        assert str(OrderStatus.OPEN) == "OrderStatus.OPEN"
        assert repr(OrderStatus.OPEN) == "<OrderStatus.OPEN: 'open'>"


class TestOrder:
    """Test cases for Order dataclass."""

    def test_order_creation_minimal(self):
        """Test creating Order with minimal required fields."""
        order = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )

        assert order.id == "test_123"
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET
        assert order.side == "buy"
        assert order.amount == Decimal("1.0")

        # Test default values
        assert order.price is None
        assert order.status == OrderStatus.OPEN
        assert order.filled == Decimal(0)
        assert order.remaining == Decimal(0)
        assert order.cost == Decimal(0)
        assert order.fee is None
        assert order.trailing_stop is None
        assert order.timestamp == 0
        assert order.params is None

    def test_order_creation_complete(self):
        """Test creating Order with all fields specified."""
        order = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side="sell",
            amount=Decimal("2.5"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("2.5"),
            remaining=Decimal("0"),
            cost=Decimal("125000"),
            fee={"cost": 2.5, "currency": "USDT"},
            trailing_stop=Decimal("49500"),
            timestamp=1234567890,
            params={"test_param": "value"}
        )

        assert order.id == "test_123"
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.LIMIT
        assert order.side == "sell"
        assert order.amount == Decimal("2.5")
        assert order.price == Decimal("50000")
        assert order.status == OrderStatus.FILLED
        assert order.filled == Decimal("2.5")
        assert order.remaining == Decimal("0")
        assert order.cost == Decimal("125000")
        assert order.fee == {"cost": 2.5, "currency": "USDT"}
        assert order.trailing_stop == Decimal("49500")
        assert order.timestamp == 1234567890
        assert order.params == {"test_param": "value"}

    def test_order_equality(self):
        """Test Order equality comparison."""
        order1 = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )

        order2 = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )

        order3 = Order(
            id="test_456",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )

        assert order1 == order2
        assert order1 != order3

    def test_order_hash(self):
        """Test Order cannot be hashed due to mutable fields (expected behavior)."""
        order = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )

        # Order dataclass is not hashable due to mutable fields (dict, list types)
        with pytest.raises(TypeError, match="unhashable type"):
            order_set = {order}

    def test_order_string_representation(self):
        """Test Order string representations."""
        order = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )

        str_repr = str(order)
        assert "Order(" in str_repr
        assert "id='test_123'" in str_repr
        assert "symbol='BTC/USDT'" in str_repr

    def test_order_field_types(self):
        """Test Order field types are not enforced at runtime (dataclass behavior)."""
        # Note: Dataclasses don't enforce type hints at runtime by default
        # This test verifies that incorrect types are accepted (expected behavior)

        # Test that string can be passed instead of Decimal (not enforced)
        order1 = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount="1.0"  # String instead of Decimal - allowed by dataclass
        )
        assert order1.amount == "1.0"  # String is stored as-is

        # Test that string can be passed for filled field
        order2 = Order(
            id="test_456",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            filled="1.0"  # String instead of Decimal - allowed by dataclass
        )
        assert order2.filled == "1.0"  # String is stored as-is

    def test_order_immutability(self):
        """Test that Order fields can be modified (dataclass is mutable by default)."""
        order = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )

        # Should be able to modify fields
        order.status = OrderStatus.FILLED
        order.filled = Decimal("1.0")
        order.cost = Decimal("50000")

        assert order.status == OrderStatus.FILLED
        assert order.filled == Decimal("1.0")
        assert order.cost == Decimal("50000")

    def test_order_with_none_values(self):
        """Test Order creation with None values for optional fields."""
        order = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=None,
            fee=None,
            trailing_stop=None,
            params=None
        )

        assert order.price is None
        assert order.fee is None
        assert order.trailing_stop is None
        assert order.params is None

    def test_order_with_complex_params(self):
        """Test Order with complex parameter structures."""
        complex_params = {
            "stop_loss": "49000",
            "take_profit": "51000",
            "trailing_stop": {"distance": "1000"},
            "metadata": {"source": "test", "version": "1.0"}
        }

        order = Order(
            id="test_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            params=complex_params
        )

        assert order.params == complex_params
        assert order.params["stop_loss"] == "49000"
        assert order.params["trailing_stop"]["distance"] == "1000"


class TestTypeImports:
    """Test that types can be imported from different locations."""

    def test_import_from_core_types(self):
        """Test importing types from core.types module."""
        from core.types import TradingMode as TM1, OrderType as OT1, OrderStatus as OS1, Order as O1

        # Verify they are the same classes
        assert TM1 is TradingMode
        assert OT1 is OrderType
        assert OS1 is OrderStatus
        assert O1 is Order

    def test_import_from_core_types_package(self):
        """Test importing types from core.types package."""
        from core.types import TradingMode as TM2, OrderType as OT2, OrderStatus as OS2, Order as O2

        # Verify they are the same classes
        assert TM2 is TradingMode
        assert OT2 is OrderType
        assert OS2 is OrderStatus
        assert O2 is Order

    def test_backward_compatibility(self):
        """Test that old imports still work."""
        # This should work without errors
        from core.types import TradingMode, OrderType, OrderStatus, Order

        assert TradingMode.LIVE.value == "live"
        assert OrderType.MARKET.value == "market"
        assert OrderStatus.OPEN.value == "open"

        # Test creating an Order instance
        order = Order(
            id="compat_test",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0")
        )
        assert order.id == "compat_test"
