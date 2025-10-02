"""
Data models for the crypto trading bot API.
Includes both SQLAlchemy database models and Pydantic validation models.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core.exceptions import SchemaValidationError
from core.logging_utils import LogSensitivity, get_structured_logger

logger = get_structured_logger("api.models", LogSensitivity.SECURE)

Base = declarative_base()

# Database URL - use environment variable or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading_bot.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
    if DATABASE_URL.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Order(Base):
    """Model for storing trade orders."""

    __tablename__ = "orders"

    id = Column(String, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    side = Column(String)  # buy/sell
    quantity = Column(Float)
    price = Column(Float)
    pnl = Column(Float, default=0.0)
    equity = Column(Float)
    cumulative_return = Column(Float, default=0.0)


class Signal(Base):
    """Model for storing trading signals."""

    __tablename__ = "signals"

    id = Column(String, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    signal_type = Column(String)  # buy/sell/hold
    strategy = Column(String)


class Equity(Base):
    """Model for storing equity progression."""

    __tablename__ = "equity"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    balance = Column(Float)
    equity = Column(Float)
    cumulative_return = Column(Float, default=0.0)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def init_db():
    """Initialize database and create tables."""
    create_tables()


# Initialize database on import
init_db()


# Pydantic validation models for inbound data
# These models enforce strict schema validation on all external data


class TickerPayload(BaseModel):
    """
    Schema for market ticker data from exchanges.

    Validates ticker information including symbol, price, and timestamp.
    """

    model_config = {"strict": True, "extra": "forbid"}

    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTC/USDT')")
    price: float = Field(..., gt=0, description="Current price in quote currency")
    timestamp: float = Field(..., ge=0, description="Unix timestamp in seconds")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format."""
        if not v or not isinstance(v, str):
            raise ValueError("Symbol must be a non-empty string")
        # Basic validation - could be extended for specific exchange formats
        if len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty or whitespace")
        return v.strip().upper()

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price is positive and reasonable."""
        if not isinstance(v, (int, float)):
            raise ValueError("Price must be a number")
        if v <= 0:
            raise ValueError("Price must be positive")
        # Could add upper bounds for sanity checking
        return float(v)


class OrderBookPayload(BaseModel):
    """
    Schema for order book data from exchanges.

    Validates bid/ask order book information.
    """

    model_config = {"strict": True, "extra": "forbid"}

    symbol: str = Field(..., description="Trading pair symbol")
    bids: List[List[float]] = Field(
        ..., description="Bid orders: [[price, quantity], ...]"
    )
    asks: List[List[float]] = Field(
        ..., description="Ask orders: [[price, quantity], ...]"
    )
    timestamp: float = Field(..., ge=0, description="Unix timestamp in seconds")

    @field_validator("bids", "asks")
    @classmethod
    def validate_order_book(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate order book entries."""
        if not isinstance(v, list):
            raise ValueError("Order book must be a list")

        for i, entry in enumerate(v):
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError(f"Order book entry {i} must be [price, quantity]")

            price, quantity = entry
            if not isinstance(price, (int, float)) or price <= 0:
                raise ValueError(f"Invalid price in order book entry {i}")
            if not isinstance(quantity, (int, float)) or quantity < 0:
                raise ValueError(f"Invalid quantity in order book entry {i}")

        return v


class TradePayload(BaseModel):
    """
    Schema for individual trade data from exchanges.

    Validates trade execution information.
    """

    model_config = {"strict": True, "extra": "forbid"}

    symbol: str = Field(..., description="Trading pair symbol")
    price: float = Field(..., gt=0, description="Trade price")
    quantity: float = Field(..., gt=0, description="Trade quantity")
    timestamp: float = Field(..., ge=0, description="Unix timestamp in seconds")
    side: str = Field(..., description="Trade side: 'buy' or 'sell'")
    trade_id: str = Field(..., description="Unique trade identifier")

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        """Validate trade side."""
        if v.lower() not in ["buy", "sell"]:
            raise ValueError("Side must be 'buy' or 'sell'")
        return v.lower()


class MarketDataPayload(BaseModel):
    """
    Schema for general market data payloads.

    Flexible schema that can contain various market data types.
    """

    model_config = {"strict": True, "extra": "forbid"}

    symbol: str = Field(..., description="Trading pair symbol")
    data_type: str = Field(..., description="Type of market data")
    payload: Dict[str, Any] = Field(..., description="Market data payload")
    timestamp: float = Field(..., ge=0, description="Unix timestamp in seconds")


class WebSocketMessagePayload(BaseModel):
    """
    Schema for WebSocket messages from exchanges.

    Validates WebSocket message structure.
    """

    model_config = {"strict": True, "extra": "forbid"}

    event_type: str = Field(..., description="WebSocket event type")
    data: Dict[str, Any] = Field(..., description="Event data payload")
    timestamp: Optional[float] = Field(
        None, ge=0, description="Unix timestamp in seconds"
    )


# Schema validation functions


def validate_ticker_data(raw_data: Dict[str, Any]) -> TickerPayload:
    """
    Validate and parse ticker data.

    Args:
        raw_data: Raw ticker data from exchange

    Returns:
        Validated TickerPayload instance

    Raises:
        SchemaValidationError: If validation fails
    """
    try:
        return TickerPayload.model_validate(raw_data)
    except ValidationError as e:
        logger.error(f"Malformed ticker payload rejected: {raw_data} -> {e}")
        raise SchemaValidationError(
            f"Invalid ticker data: {e}",
            data=raw_data,
            schema_name="TickerPayload",
            field_errors=e.errors(),
        )


def validate_order_book_data(raw_data: Dict[str, Any]) -> OrderBookPayload:
    """
    Validate and parse order book data.

    Args:
        raw_data: Raw order book data from exchange

    Returns:
        Validated OrderBookPayload instance

    Raises:
        SchemaValidationError: If validation fails
    """
    try:
        return OrderBookPayload.model_validate(raw_data)
    except ValidationError as e:
        logger.error(f"Malformed order book payload rejected: {raw_data} -> {e}")
        raise SchemaValidationError(
            f"Invalid order book data: {e}",
            data=raw_data,
            schema_name="OrderBookPayload",
            field_errors=e.errors(),
        )


def validate_trade_data(raw_data: Dict[str, Any]) -> TradePayload:
    """
    Validate and parse trade data.

    Args:
        raw_data: Raw trade data from exchange

    Returns:
        Validated TradePayload instance

    Raises:
        SchemaValidationError: If validation fails
    """
    try:
        return TradePayload.model_validate(raw_data)
    except ValidationError as e:
        logger.error(f"Malformed trade payload rejected: {raw_data} -> {e}")
        raise SchemaValidationError(
            f"Invalid trade data: {e}",
            data=raw_data,
            schema_name="TradePayload",
            field_errors=e.errors(),
        )


def validate_market_data(raw_data: Dict[str, Any]) -> MarketDataPayload:
    """
    Validate and parse general market data.

    Args:
        raw_data: Raw market data from exchange

    Returns:
        Validated MarketDataPayload instance

    Raises:
        SchemaValidationError: If validation fails
    """
    try:
        return MarketDataPayload.model_validate(raw_data)
    except ValidationError as e:
        logger.error(f"Malformed market data payload rejected: {raw_data} -> {e}")
        raise SchemaValidationError(
            f"Invalid market data: {e}",
            data=raw_data,
            schema_name="MarketDataPayload",
            field_errors=e.errors(),
        )


def validate_websocket_message(raw_data: Dict[str, Any]) -> WebSocketMessagePayload:
    """
    Validate and parse WebSocket message data.

    Args:
        raw_data: Raw WebSocket message from exchange

    Returns:
        Validated WebSocketMessagePayload instance

    Raises:
        SchemaValidationError: If validation fails
    """
    try:
        return WebSocketMessagePayload.model_validate(raw_data)
    except ValidationError as e:
        logger.error(f"Malformed WebSocket message rejected: {raw_data} -> {e}")
        raise SchemaValidationError(
            f"Invalid WebSocket message: {e}",
            data=raw_data,
            schema_name="WebSocketMessagePayload",
            field_errors=e.errors(),
        )
