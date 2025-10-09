"""
tests/test_schema_validation.py

Tests for strict schema validation of inbound market and API data.
"""

import pytest
from pydantic import ValidationError

from api.models import (
    MarketDataPayload,
    OrderBookPayload,
    TickerPayload,
    TradePayload,
    WebSocketMessagePayload,
    validate_market_data,
    validate_order_book_data,
    validate_ticker_data,
    validate_trade_data,
    validate_websocket_message,
)
from core.exceptions import SchemaValidationError


class TestTickerPayloadValidation:
    """Test validation of ticker data payloads."""

    def test_valid_ticker_payload(self):
        """Test that valid ticker data is accepted."""
        data = {"symbol": "BTC/USDT", "price": 45000.50, "timestamp": 1640995200.0}

        payload = TickerPayload.model_validate(data)
        assert payload.symbol == "BTC/USDT"
        assert payload.price == 45000.50
        assert payload.timestamp == 1640995200.0

    def test_ticker_symbol_validation(self):
        """Test symbol validation in ticker data."""
        # Valid symbol
        data = {"symbol": "ETH/USD", "price": 3000.0, "timestamp": 1640995200.0}
        payload = TickerPayload.model_validate(data)
        assert payload.symbol == "ETH/USD"

        # Symbol gets uppercased
        data["symbol"] = "btc/usdt"
        payload = TickerPayload.model_validate(data)
        assert payload.symbol == "BTC/USDT"

    def test_ticker_price_validation(self):
        """Test price validation in ticker data."""
        # Valid positive price
        data = {"symbol": "BTC/USDT", "price": 50000.0, "timestamp": 1640995200.0}
        payload = TickerPayload.model_validate(data)
        assert payload.price == 50000.0

        # Zero price should fail
        data["price"] = 0
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

        # Negative price should fail
        data["price"] = -1000
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

    def test_missing_required_field(self):
        """Test that missing required fields are rejected."""
        # Missing symbol
        data = {"price": 45000.50, "timestamp": 1640995200.0}
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

        # Missing price
        data = {"symbol": "BTC/USDT", "timestamp": 1640995200.0}
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

        # Missing timestamp
        data = {"symbol": "BTC/USDT", "price": 45000.50}
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

    def test_extra_fields_rejected_in_strict_mode(self):
        """Test that extra fields are rejected in strict mode."""
        data = {
            "symbol": "BTC/USDT",
            "price": 45000.50,
            "timestamp": 1640995200.0,
            "extra_field": "should_be_rejected",
        }
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

    def test_wrong_data_type(self):
        """Test that wrong data types are rejected."""
        # String price instead of number
        data = {"symbol": "BTC/USDT", "price": "45000.50", "timestamp": 1640995200.0}
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

        # Negative timestamp
        data = {"symbol": "BTC/USDT", "price": 45000.50, "timestamp": -1}
        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)


class TestOrderBookPayloadValidation:
    """Test validation of order book data payloads."""

    def test_valid_order_book_payload(self):
        """Test that valid order book data is accepted."""
        data = {
            "symbol": "BTC/USDT",
            "bids": [[45000.0, 1.5], [44950.0, 2.0]],
            "asks": [[45050.0, 1.0], [45100.0, 0.5]],
            "timestamp": 1640995200.0,
        }

        payload = OrderBookPayload.model_validate(data)
        assert payload.symbol == "BTC/USDT"
        assert len(payload.bids) == 2
        assert len(payload.asks) == 2
        assert payload.bids[0] == [45000.0, 1.5]

    def test_invalid_order_book_entry(self):
        """Test that invalid order book entries are rejected."""
        # Wrong structure (not a list of lists)
        data = {
            "symbol": "BTC/USDT",
            "bids": [[45000.0], [44950.0, 2.0]],  # Missing quantity
            "asks": [[45050.0, 1.0]],
            "timestamp": 1640995200.0,
        }
        with pytest.raises(ValidationError):
            OrderBookPayload.model_validate(data)

        # Negative price
        data = {
            "symbol": "BTC/USDT",
            "bids": [[-45000.0, 1.5]],
            "asks": [[45050.0, 1.0]],
            "timestamp": 1640995200.0,
        }
        with pytest.raises(ValidationError):
            OrderBookPayload.model_validate(data)

        # Negative quantity
        data = {
            "symbol": "BTC/USDT",
            "bids": [[45000.0, -1.5]],
            "asks": [[45050.0, 1.0]],
            "timestamp": 1640995200.0,
        }
        with pytest.raises(ValidationError):
            OrderBookPayload.model_validate(data)


class TestTradePayloadValidation:
    """Test validation of trade data payloads."""

    def test_valid_trade_payload(self):
        """Test that valid trade data is accepted."""
        data = {
            "symbol": "BTC/USDT",
            "price": 45000.0,
            "quantity": 0.5,
            "timestamp": 1640995200.0,
            "side": "buy",
            "trade_id": "123456789",
        }

        payload = TradePayload.model_validate(data)
        assert payload.symbol == "BTC/USDT"
        assert payload.price == 45000.0
        assert payload.quantity == 0.5
        assert payload.side == "buy"
        assert payload.trade_id == "123456789"

    def test_trade_side_validation(self):
        """Test trade side validation."""
        # Valid sides
        for side in ["buy", "sell", "BUY", "SELL"]:
            data = {
                "symbol": "BTC/USDT",
                "price": 45000.0,
                "quantity": 0.5,
                "timestamp": 1640995200.0,
                "side": side,
                "trade_id": "123",
            }
            payload = TradePayload.model_validate(data)
            assert payload.side in ["buy", "sell"]

        # Invalid side
        data["side"] = "invalid"
        with pytest.raises(ValidationError):
            TradePayload.model_validate(data)


class TestValidationFunctions:
    """Test the validation helper functions."""

    def test_validate_ticker_data_success(self):
        """Test successful ticker data validation."""
        data = {"symbol": "BTC/USDT", "price": 45000.50, "timestamp": 1640995200.0}

        result = validate_ticker_data(data)
        assert isinstance(result, TickerPayload)
        assert result.symbol == "BTC/USDT"

    def test_validate_ticker_data_failure(self):
        """Test ticker data validation failure."""
        data = {
            "symbol": "BTC/USDT",
            "price": "invalid_price",  # Wrong type
            "timestamp": 1640995200.0,
        }

        with pytest.raises(SchemaValidationError) as exc_info:
            validate_ticker_data(data)

        assert "Invalid ticker data" in str(exc_info.value)
        assert exc_info.value.schema_name == "TickerPayload"
        assert exc_info.value.data == data

    def test_validate_order_book_data_success(self):
        """Test successful order book data validation."""
        data = {
            "symbol": "BTC/USDT",
            "bids": [[45000.0, 1.5]],
            "asks": [[45050.0, 1.0]],
            "timestamp": 1640995200.0,
        }

        result = validate_order_book_data(data)
        assert isinstance(result, OrderBookPayload)
        assert result.symbol == "BTC/USDT"

    def test_validate_trade_data_success(self):
        """Test successful trade data validation."""
        data = {
            "symbol": "BTC/USDT",
            "price": 45000.0,
            "quantity": 0.5,
            "timestamp": 1640995200.0,
            "side": "buy",
            "trade_id": "123",
        }

        result = validate_trade_data(data)
        assert isinstance(result, TradePayload)
        assert result.side == "buy"

    def test_validate_market_data_success(self):
        """Test successful market data validation."""
        data = {
            "symbol": "BTC/USDT",
            "data_type": "ticker",
            "payload": {"price": 45000.0},
            "timestamp": 1640995200.0,
        }

        result = validate_market_data(data)
        assert isinstance(result, MarketDataPayload)
        assert result.data_type == "ticker"

    def test_validate_websocket_message_success(self):
        """Test successful WebSocket message validation."""
        data = {
            "event_type": "ticker",
            "data": {"symbol": "BTC/USDT", "price": 45000.0},
        }

        result = validate_websocket_message(data)
        assert isinstance(result, WebSocketMessagePayload)
        assert result.event_type == "ticker"


class TestSchemaValidationError:
    """Test the custom SchemaValidationError exception."""

    def test_error_creation(self):
        """Test SchemaValidationError creation."""
        error = SchemaValidationError("Test message")
        assert str(error) == "Test message"

    def test_error_with_schema_name(self):
        """Test error with schema name."""
        error = SchemaValidationError("Validation failed", schema_name="TickerPayload")
        assert "[TickerPayload] Validation failed" in str(error)

    def test_error_with_field_errors(self):
        """Test error with field errors."""
        field_errors = {"price": ["must be positive"]}
        error = SchemaValidationError("Validation failed", field_errors=field_errors)
        assert "field errors:" in str(error)

    def test_error_with_all_fields(self):
        """Test error with all fields."""
        field_errors = {"symbol": ["required"]}
        error = SchemaValidationError(
            "Validation failed",
            data={"price": 45000},
            schema_name="TickerPayload",
            field_errors=field_errors,
        )
        error_str = str(error)
        assert "[TickerPayload] Validation failed" in error_str
        assert "field errors:" in error_str
        assert error.data == {"price": 45000}
        assert error.field_errors == field_errors


class TestCoercionBehavior:
    """Test type coercion behavior in strict mode."""

    def test_strict_mode_rejects_type_mismatches(self):
        """Test that strict mode rejects type mismatches."""
        # String that could be converted to float
        data = {
            "symbol": "BTC/USDT",
            "price": "45000.50",  # String instead of float
            "timestamp": 1640995200.0,
        }

        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)

    def test_strict_mode_rejects_extra_fields(self):
        """Test that strict mode rejects extra fields."""
        data = {
            "symbol": "BTC/USDT",
            "price": 45000.50,
            "timestamp": 1640995200.0,
            "extra_field": "not allowed",
        }

        with pytest.raises(ValidationError):
            TickerPayload.model_validate(data)
