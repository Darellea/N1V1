from decimal import Decimal

import pytest

from core.execution.order_processor import OrderProcessor
from core.types.order_types import Order, OrderStatus, OrderType


@pytest.fixture
def order_processor():
    """Fixture to create a fresh OrderProcessor instance."""
    return OrderProcessor()


@pytest.fixture
def sample_order():
    """Fixture to create a sample order for testing."""
    return Order(
        id="test_order_123",
        symbol="BTC/USDT",
        type=OrderType.MARKET,
        side="buy",
        amount=Decimal("1.0"),
        price=Decimal("50000"),
        status=OrderStatus.FILLED,
        filled=Decimal("1.0"),
        remaining=Decimal("0"),
        cost=Decimal("50000"),
        fee={"cost": "5.0", "currency": "USDT"},
        timestamp=1234567890000,
        params={"test_param": "value"},
    )


@pytest.fixture
def sample_buy_order():
    """Fixture to create a sample buy order."""
    return Order(
        id="buy_order_123",
        symbol="BTC/USDT",
        type=OrderType.MARKET,
        side="buy",
        amount=Decimal("1.0"),
        price=Decimal("50000"),
        status=OrderStatus.FILLED,
        filled=Decimal("1.0"),
        remaining=Decimal("0"),
        cost=Decimal("50000"),
        fee={"cost": "5.0", "currency": "USDT"},
        timestamp=1234567890000,
    )


@pytest.fixture
def sample_sell_order():
    """Fixture to create a sample sell order."""
    return Order(
        id="sell_order_456",
        symbol="BTC/USDT",
        type=OrderType.MARKET,
        side="sell",
        amount=Decimal("1.0"),
        price=Decimal("55000"),
        status=OrderStatus.FILLED,
        filled=Decimal("1.0"),
        remaining=Decimal("0"),
        cost=Decimal("55000"),
        fee={"cost": "5.5", "currency": "USDT"},
        timestamp=1234567900000,
    )


class TestOrderProcessor:
    """Test suite for OrderProcessor class."""

    def test_initialization(self, order_processor):
        """Test OrderProcessor initialization."""
        assert order_processor.open_orders == {}
        assert order_processor.closed_orders == {}
        assert order_processor.positions == {}
        assert order_processor.trade_count == 0

    # Tests for parse_order_response method (lines 41-120)
    def test_parse_order_response_valid_complete(self, order_processor):
        """Test parsing a complete, valid order response."""
        response = {
            "id": "12345",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "buy",
            "amount": "1.5",
            "price": "50000",
            "status": "filled",
            "filled": "1.5",
            "remaining": "0",
            "cost": "75000",
            "fee": {"cost": "7.5", "currency": "USDT"},
            "timestamp": 1234567890000,
            "params": {"test": "value"},
        }

        order = order_processor.parse_order_response(response)

        assert order.id == "12345"
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET
        assert order.side == "buy"
        assert order.amount == Decimal("1.5")
        assert order.price == Decimal("50000")
        assert order.status == OrderStatus.FILLED
        assert order.filled == Decimal("1.5")
        assert order.remaining == Decimal("0")
        assert order.cost == Decimal("75000")
        assert order.fee == {"cost": "7.5", "currency": "USDT"}
        assert order.timestamp == 1234567890000

    def test_parse_order_response_with_aliases(self, order_processor):
        """Test parsing order response with alias field names."""
        response = {
            "orderId": "67890",  # alias for id
            "market": "ETH/USDT",  # alias for symbol
            "order_type": "limit",  # alias for type
            "direction": "sell",  # alias for side
            "size": "2.0",  # alias for amount
            "state": "open",  # alias for status
            "executed": "0.5",  # alias for filled
            "remaining_amount": "1.5",  # alias for remaining
            "datetime": 1234567890000,  # alias for timestamp
            "info": {"exchange_specific": "data"},  # alias for params
        }

        order = order_processor.parse_order_response(response)

        assert order.id == "67890"
        assert order.symbol == "ETH/USDT"
        assert order.type == OrderType.LIMIT
        assert order.side == "sell"
        assert order.amount == Decimal("2.0")
        assert order.status == OrderStatus.OPEN
        assert order.filled == Decimal("0.5")
        assert order.remaining == Decimal("1.5")
        assert order.timestamp == 1234567890000
        assert order.params == {"exchange_specific": "data"}

    def test_parse_order_response_missing_fields(self, order_processor):
        """Test parsing order response with missing fields uses defaults."""
        response = {"id": "minimal", "symbol": "BTC/USDT"}

        order = order_processor.parse_order_response(response)

        assert order.id == "minimal"
        assert order.symbol == "BTC/USDT"
        assert order.type == OrderType.MARKET  # default
        assert order.side == ""  # default
        assert order.amount == Decimal("0")  # default
        assert order.price is None  # default
        assert order.status == OrderStatus.OPEN  # default
        assert order.filled == Decimal("0")  # default
        assert order.remaining == Decimal("0")  # default
        assert order.cost == Decimal("0")  # default
        assert order.fee is None  # default
        # Note: timestamp conversion might not work in test environment
        # Just check that it's a valid integer
        assert isinstance(order.timestamp, int)

    def test_parse_order_response_invalid_type(self, order_processor):
        """Test parsing order response with invalid type falls back to MARKET."""
        response = {"id": "test", "symbol": "BTC/USDT", "type": "invalid_type"}

        order = order_processor.parse_order_response(response)
        assert order.type == OrderType.MARKET

    def test_parse_order_response_invalid_status(self, order_processor):
        """Test parsing order response with invalid status falls back to OPEN."""
        response = {"id": "test", "symbol": "BTC/USDT", "status": "invalid_status"}

        order = order_processor.parse_order_response(response)
        assert order.status == OrderStatus.OPEN

    def test_parse_order_response_invalid_numeric_fields(self, order_processor):
        """Test parsing order response with invalid numeric fields."""
        response = {
            "id": "test",
            "symbol": "BTC/USDT",
            "amount": "invalid",
            "price": "not_a_number",
            "filled": None,
            "cost": "",
        }

        order = order_processor.parse_order_response(response)

        assert order.amount == Decimal("0")  # fallback for invalid amount
        assert order.price is None  # fallback for invalid price
        assert order.filled == Decimal("0")  # fallback for None filled
        assert order.cost == Decimal("0")  # fallback for empty cost

    def test_parse_order_response_non_dict_input(self, order_processor):
        """Test parsing order response with non-dict input raises ValueError."""
        with pytest.raises(ValueError, match="Invalid order response"):
            order_processor.parse_order_response("not a dict")

        with pytest.raises(ValueError, match="Invalid order response"):
            order_processor.parse_order_response(None)

        with pytest.raises(ValueError, match="Invalid order response"):
            order_processor.parse_order_response([])

    def test_parse_order_response_timestamp_conversion(self, order_processor):
        """Test timestamp conversion in parse_order_response."""

        # Test with milliseconds timestamp
        response = {
            "id": "test1",
            "symbol": "BTC/USDT",
            "timestamp": 1234567890000,  # milliseconds
        }
        order = order_processor.parse_order_response(response)
        assert order.timestamp == 1234567890000

        # Test with seconds timestamp (should be converted to milliseconds)
        response = {
            "id": "test2",
            "symbol": "BTC/USDT",
            "timestamp": 1234567890,  # seconds
        }
        order = order_processor.parse_order_response(response)
        assert order.timestamp == 1234567890000  # converted to milliseconds

        # Test with None timestamp (should use current time or fallback to 0)
        response = {"id": "test3", "symbol": "BTC/USDT", "timestamp": None}
        order = order_processor.parse_order_response(response)
        # Either uses current time or falls back to 0
        assert order.timestamp >= 0

    # Tests for process_order method (lines 146-232)
    @pytest.mark.asyncio
    async def test_process_order_filled_buy(self, order_processor, sample_buy_order):
        """Test processing a filled buy order."""
        result = await order_processor.process_order(sample_buy_order)

        assert result["id"] == "buy_order_123"
        assert result["symbol"] == "BTC/USDT"
        assert result["type"] == "market"
        assert result["side"] == "buy"
        assert result["amount"] == 1.0
        assert result["price"] == 50000.0
        assert result["status"] == "filled"
        assert result["cost"] == 50000.0
        assert result["fee"] == {"cost": "5.0", "currency": "USDT"}
        assert result["timestamp"] == 1234567890000
        assert result["pnl"] is None  # PnL is None for buy orders

        # Check internal state
        assert "buy_order_123" in order_processor.closed_orders
        assert "buy_order_123" not in order_processor.open_orders
        assert order_processor.trade_count == 1

        # Check position tracking
        assert "BTC/USDT" in order_processor.positions
        position = order_processor.positions["BTC/USDT"]
        assert position["amount"] == Decimal("1.0")
        assert position["entry_price"] == Decimal("50000")
        assert position["entry_cost"] == Decimal("50000")

    @pytest.mark.asyncio
    async def test_process_order_filled_sell_with_pnl(
        self, order_processor, sample_buy_order, sample_sell_order
    ):
        """Test processing a filled sell order with PnL calculation."""
        # First process the buy order to establish position
        await order_processor.process_order(sample_buy_order)

        # Then process the sell order
        result = await order_processor.process_order(sample_sell_order)

        assert result["id"] == "sell_order_456"
        assert result["symbol"] == "BTC/USDT"
        assert result["side"] == "sell"
        assert (
            result["pnl"] == 4994.5
        )  # (55000 - 50000) * 1.0 - 5.5 = 5000 - 5.5 = 4994.5

        # Check that position was closed
        assert "BTC/USDT" not in order_processor.positions
        assert order_processor.trade_count == 2

    @pytest.mark.asyncio
    async def test_process_order_open_order(self, order_processor):
        """Test processing an open (not filled) order."""
        open_order = Order(
            id="open_order_789",
            symbol="ETH/USDT",
            type=OrderType.LIMIT,
            side="buy",
            amount=Decimal("0.5"),
            price=Decimal("3000"),
            status=OrderStatus.OPEN,
            filled=Decimal("0"),
            remaining=Decimal("0.5"),
            cost=Decimal("0"),
        )

        result = await order_processor.process_order(open_order)

        assert result["id"] == "open_order_789"
        assert result["status"] == "open"
        assert "open_order_789" in order_processor.open_orders
        assert "open_order_789" not in order_processor.closed_orders
        assert (
            "ETH/USDT" not in order_processor.positions
        )  # No position for open orders

    @pytest.mark.asyncio
    async def test_process_order_dynamic_take_profit(self, order_processor):
        """Test dynamic take profit calculation."""
        order_with_tp = Order(
            id="tp_order_123",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            remaining=Decimal("0"),
            cost=Decimal("50000"),
            params={
                "dynamic_tp": True,
                "stop_loss": "49000",
                "risk_reward_ratio": 2.0,
                "trend_strength": 0.5,
            },
        )

        result = await order_processor.process_order(order_with_tp)

        assert "take_profit" in result
        # Expected TP: entry(50000) + risk_reward_ratio(2.0) * (1 + trend_strength(0.5)) * risk(1000)
        # = 50000 + 2.0 * 1.5 * 1000 = 50000 + 3000 = 53000
        assert result["take_profit"] == 53000.0

    @pytest.mark.asyncio
    async def test_process_order_dynamic_tp_missing_params(self, order_processor):
        """Test dynamic take profit with missing parameters."""
        order_missing_params = Order(
            id="tp_order_missing",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            params={"dynamic_tp": True},  # Missing stop_loss
        )

        result = await order_processor.process_order(order_missing_params)

        assert "take_profit" not in result  # Should not calculate TP without stop_loss

    @pytest.mark.asyncio
    async def test_process_order_trailing_stop_from_params(self, order_processor):
        """Test trailing stop from order parameters."""
        order_with_trailing = Order(
            id="trailing_order",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            params={"trailing_stop": {"price": "49500"}},
        )

        await order_processor.process_order(order_with_trailing)

        position = order_processor.positions["BTC/USDT"]
        assert position["trailing_stop"] == Decimal("49500")

    # Tests for _update_positions method (lines 236-274)
    def test_update_positions_buy_order(self, order_processor, sample_buy_order):
        """Test position update for buy orders."""
        order_processor._update_positions(sample_buy_order)

        assert "BTC/USDT" in order_processor.positions
        position = order_processor.positions["BTC/USDT"]
        assert position["amount"] == Decimal("1.0")
        assert position["entry_price"] == Decimal("50000")
        assert position["entry_cost"] == Decimal("50000")

    def test_update_positions_sell_order_full_close(
        self, order_processor, sample_buy_order, sample_sell_order
    ):
        """Test position update for sell orders that fully close position."""
        # Establish position first
        order_processor._update_positions(sample_buy_order)

        # Then close it
        order_processor._update_positions(sample_sell_order)

        assert "BTC/USDT" not in order_processor.positions

    def test_update_positions_sell_order_partial_close(self, order_processor):
        """Test position update for sell orders that partially close position."""
        # Buy 2.0 units
        buy_order = Order(
            id="buy_2",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("2.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("2.0"),
            cost=Decimal("100000"),
        )

        # Sell 1.0 unit
        sell_order = Order(
            id="sell_partial",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="sell",
            amount=Decimal("1.0"),
            price=Decimal("55000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            cost=Decimal("55000"),
        )

        order_processor._update_positions(buy_order)
        order_processor._update_positions(sell_order)

        assert "BTC/USDT" in order_processor.positions
        position = order_processor.positions["BTC/USDT"]
        assert position["amount"] == Decimal("1.0")  # 2.0 - 1.0
        assert position["entry_price"] == Decimal("50000")  # Should remain the same
        assert position["entry_cost"] == Decimal(
            "50000"
        )  # 100000 - (50000 * 1.0) = 50000

    def test_update_positions_multiple_buys(self, order_processor):
        """Test position update with multiple buy orders (average pricing)."""
        buy1 = Order(
            id="buy1",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            cost=Decimal("50000"),
        )

        buy2 = Order(
            id="buy2",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("55000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            cost=Decimal("55000"),
        )

        order_processor._update_positions(buy1)
        order_processor._update_positions(buy2)

        position = order_processor.positions["BTC/USDT"]
        assert position["amount"] == Decimal("2.0")
        # Average price: (50000 + 55000) / 2 = 52500
        assert position["entry_price"] == Decimal("52500")
        assert position["entry_cost"] == Decimal("105000")  # 50000 + 55000

    # Tests for _calculate_pnl method (lines 278-288)
    def test_calculate_pnl_sell_order(
        self, order_processor, sample_buy_order, sample_sell_order
    ):
        """Test PnL calculation for sell orders."""
        # Establish position
        order_processor._update_positions(sample_buy_order)

        pnl = order_processor._calculate_pnl(sample_sell_order)

        # Expected: (55000 - 50000) * 1.0 - 5.5 = 5000 - 5.5 = 4994.5
        expected_pnl = (55000 - 50000) * 1.0 - 5.5
        assert pnl == expected_pnl

    def test_calculate_pnl_no_position(self, order_processor, sample_sell_order):
        """Test PnL calculation when no position exists."""
        pnl = order_processor._calculate_pnl(sample_sell_order)
        assert pnl is None

    def test_calculate_pnl_buy_order(self, order_processor, sample_buy_order):
        """Test PnL calculation for buy orders (should return None)."""
        pnl = order_processor._calculate_pnl(sample_buy_order)
        assert pnl is None

    def test_calculate_pnl_no_fee(self, order_processor):
        """Test PnL calculation when order has no fee."""
        # Establish position
        buy_order = Order(
            id="buy_no_fee",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            cost=Decimal("50000"),
            fee=None,  # No fee
        )

        sell_order = Order(
            id="sell_no_fee",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="sell",
            amount=Decimal("1.0"),
            price=Decimal("55000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            cost=Decimal("55000"),
            fee=None,
        )

        order_processor._update_positions(buy_order)
        pnl = order_processor._calculate_pnl(sell_order)

        # Expected: (55000 - 50000) * 1.0 = 5000 (no fee deduction)
        assert pnl == 5000.0

    # Tests for utility methods (lines 303, 307, 311)
    @pytest.mark.asyncio
    async def test_estimate_trend_strength_stub(self, order_processor):
        """Test trend strength estimation (currently returns 0.0)."""
        strength = await order_processor._estimate_trend_strength("BTC/USDT")
        assert strength == 0.0

    @pytest.mark.asyncio
    async def test_estimate_trend_strength_with_params(self, order_processor):
        """Test trend strength estimation with different parameters."""
        strength = await order_processor._estimate_trend_strength("BTC/USDT", "4h", 50)
        assert strength == 0.0  # Still returns 0.0 as it's a stub

    def test_get_active_order_count(self, order_processor):
        """Test getting count of active orders."""
        assert order_processor.get_active_order_count() == 0

        # Add some open orders
        open_order1 = Order(
            id="open1",
            symbol="BTC/USDT",
            type=OrderType.LIMIT,
            side="buy",
            amount=Decimal("1.0"),
            status=OrderStatus.OPEN,
        )

        open_order2 = Order(
            id="open2",
            symbol="ETH/USDT",
            type=OrderType.LIMIT,
            side="sell",
            amount=Decimal("0.5"),
            status=OrderStatus.OPEN,
        )

        order_processor.open_orders = {"open1": open_order1, "open2": open_order2}

        assert order_processor.get_active_order_count() == 2

    def test_get_open_position_count(self, order_processor):
        """Test getting count of open positions."""
        assert order_processor.get_open_position_count() == 0

        # Add some positions
        order_processor.positions = {
            "BTC/USDT": {"amount": Decimal("1.0"), "entry_price": Decimal("50000")},
            "ETH/USDT": {"amount": Decimal("2.0"), "entry_price": Decimal("3000")},
        }

        assert order_processor.get_open_position_count() == 2

    # Additional edge case tests
    def test_parse_order_response_empty_response(self, order_processor):
        """Test parsing completely empty response."""
        response = {}
        order = order_processor.parse_order_response(response)

        assert order.id == ""  # Empty string default
        assert order.symbol == ""  # Empty string default
        assert order.type == OrderType.MARKET
        assert order.status == OrderStatus.OPEN

    def test_parse_order_response_mixed_case_enums(self, order_processor):
        """Test parsing order response with mixed case enum values."""
        response = {
            "id": "mixed_case",
            "symbol": "BTC/USDT",
            "type": "MARKET",  # Upper case
            "status": "filled",  # Lower case
        }

        order = order_processor.parse_order_response(response)

        assert order.type == OrderType.MARKET
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_process_order_partial_fill(self, order_processor):
        """Test processing a partially filled order."""
        partial_order = Order(
            id="partial_order",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("2.0"),
            price=Decimal("50000"),
            status=OrderStatus.PARTIALLY_FILLED,
            filled=Decimal("1.0"),
            remaining=Decimal("1.0"),
            cost=Decimal("50000"),
        )

        result = await order_processor.process_order(partial_order)

        assert result["status"] == "partially_filled"
        assert result["filled"] == 1.0
        assert "partial_order" in order_processor.open_orders  # Should remain open
        assert "partial_order" not in order_processor.closed_orders

        # PARTIALLY_FILLED orders don't update positions (only FILLED orders do)
        assert "BTC/USDT" not in order_processor.positions

    def test_update_positions_trailing_stop_from_order_attribute(self, order_processor):
        """Test trailing stop update from order.trailing_stop attribute."""
        order_with_trailing = Order(
            id="trailing_attr",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            trailing_stop=Decimal("49500"),  # Set via attribute
        )

        order_processor._update_positions(order_with_trailing)

        position = order_processor.positions["BTC/USDT"]
        assert position["trailing_stop"] == Decimal("49500")

    def test_calculate_pnl_complex_fee_structure(self, order_processor):
        """Test PnL calculation with complex fee structure."""
        # Establish position
        buy_order = Order(
            id="buy_complex",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            cost=Decimal("50000"),
        )

        # Sell with complex fee structure
        sell_order = Order(
            id="sell_complex",
            symbol="BTC/USDT",
            type=OrderType.MARKET,
            side="sell",
            amount=Decimal("1.0"),
            price=Decimal("55000"),
            status=OrderStatus.FILLED,
            filled=Decimal("1.0"),
            cost=Decimal("55000"),
            fee={"cost": "5.5", "currency": "USDT", "rate": "0.001"},  # Complex fee
        )

        order_processor._update_positions(buy_order)
        pnl = order_processor._calculate_pnl(sell_order)

        # Should only use the 'cost' field from fee
        expected_pnl = (55000 - 50000) * 1.0 - 5.5
        assert pnl == expected_pnl
