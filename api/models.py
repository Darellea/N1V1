"""
Database models for the crypto trading bot API.
Uses SQLAlchemy with SQLite for persistence.
"""

import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

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
