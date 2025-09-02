"""
Pydantic schemas for API request/response models.
Standardizes error responses and data validation.
"""

from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, Optional


class ErrorDetail(BaseModel):
    """Error detail model for standardized error responses."""
    code: int
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: ErrorDetail

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": 401,
                    "message": "Invalid API key",
                    "details": None
                }
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    bot_running: bool

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2023-12-01T12:00:00.000000",
                "bot_running": True
            }
        }
    )


class StatusResponse(BaseModel):
    """Bot status response model."""
    running: bool
    paused: bool
    mode: str
    pairs: list
    timestamp: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "running": True,
                "paused": False,
                "mode": "LIVE",
                "pairs": ["BTC/USDT", "ETH/USDT"],
                "timestamp": "2023-12-01T12:00:00.000000"
            }
        }
    )


class OrderResponse(BaseModel):
    """Order response model."""
    id: str
    symbol: str
    timestamp: Optional[str]
    side: Optional[str]
    quantity: Optional[float]
    price: Optional[float]
    pnl: Optional[float]
    equity: Optional[float]
    cumulative_return: Optional[float]


class OrdersListResponse(BaseModel):
    """Orders list response model."""
    orders: list[OrderResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "orders": [
                    {
                        "id": "trade_123",
                        "symbol": "BTC/USDT",
                        "timestamp": "2023-12-01T12:00:00.000000",
                        "side": "buy",
                        "quantity": 0.001,
                        "price": 50000.0,
                        "pnl": 100.0,
                        "equity": 10500.0,
                        "cumulative_return": 0.05
                    }
                ]
            }
        }
    )


class SignalResponse(BaseModel):
    """Signal response model."""
    id: str
    symbol: str
    timestamp: Optional[str]
    confidence: Optional[float]
    signal_type: Optional[str]
    strategy: Optional[str]


class SignalsListResponse(BaseModel):
    """Signals list response model."""
    signals: list[SignalResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "signals": [
                    {
                        "id": "signal_123",
                        "symbol": "BTC/USDT",
                        "timestamp": "2023-12-01T12:00:00.000000",
                        "confidence": 0.85,
                        "signal_type": "buy",
                        "strategy": "RSIStrategy"
                    }
                ]
            }
        }
    )


class EquityPointResponse(BaseModel):
    """Equity point response model."""
    timestamp: Optional[str]
    balance: Optional[float]
    equity: Optional[float]
    cumulative_return: Optional[float]


class EquityCurveResponse(BaseModel):
    """Equity curve response model."""
    equity_curve: list[EquityPointResponse]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "equity_curve": [
                    {
                        "timestamp": "2023-12-01T12:00:00.000000",
                        "balance": 10000.0,
                        "equity": 10000.0,
                        "cumulative_return": 0.0
                    }
                ]
            }
        }
    )


class PerformanceResponse(BaseModel):
    """Performance metrics response model."""
    total_pnl: float
    win_rate: float
    wins: int
    losses: int
    sharpe_ratio: float
    max_drawdown: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_pnl": 500.0,
                "win_rate": 0.65,
                "wins": 13,
                "losses": 7,
                "sharpe_ratio": 1.23,
                "max_drawdown": 0.15
            }
        }
    )


class SuccessResponse(BaseModel):
    """Generic success response model."""
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Bot paused successfully"
            }
        }
    )
