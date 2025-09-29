"""
core/types/order_types.py

Order-related dataclasses and enums extracted from order_manager.py and types.py.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional


class OrderType(Enum):
    """Supported order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status states."""

    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Dataclass representing an order."""

    id: str
    symbol: str
    type: OrderType
    side: str  # 'buy' or 'sell'
    amount: Decimal
    price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.OPEN
    filled: Decimal = Decimal(0)
    remaining: Decimal = Decimal(0)
    cost: Decimal = Decimal(0)
    fee: Dict = None
    trailing_stop: Optional[Decimal] = None
    timestamp: int = 0
    params: Dict = None
