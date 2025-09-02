"""
FastAPI application for the crypto trading bot.
Provides REST endpoints for monitoring and controlling the bot.
"""

from fastapi import FastAPI, HTTPException, Request, Depends, Response, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Any, Optional
import logging
import os
from datetime import datetime
from sqlalchemy.orm import Session
from .models import Order, Signal, Equity, get_db
from .schemas import ErrorResponse
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

# Global reference to bot engine (will be set when app starts)
bot_engine = None

# API Key authentication
API_KEY = os.getenv("API_KEY")
security = HTTPBearer(auto_error=False) if API_KEY else None

app = FastAPI(
    title="Crypto Trading Bot API",
    description="REST API for monitoring and controlling the crypto trading bot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Create API router with /api/v1 prefix for versioning
api_router = APIRouter(prefix="/api/v1")

# Prometheus metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)

trades_total = Counter(
    'trades_total',
    'Total number of trades executed',
    ['symbol', 'side']
)

wins_total = Counter('wins_total', 'Total number of winning trades')
losses_total = Counter('losses_total', 'Total number of losing trades')
open_positions = Counter('open_positions', 'Current number of open positions')


async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key if authentication is enabled."""
    if API_KEY and credentials:
        if credentials.credentials != API_KEY:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "code": 401,
                        "message": "Invalid API key"
                    }
                }
            )
    elif API_KEY and not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": 401,
                    "message": "API key required"
                }
            }
        )
    return True


def set_bot_engine(engine):
    """Set the global bot engine reference."""
    global bot_engine
    bot_engine = engine


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Crypto Trading Bot API", "version": "1.0.0"}


@api_router.get("/status")
async def get_status():
    """Get bot status."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": 503,
                    "message": "Bot engine not available"
                }
            }
        )

    try:
        return {
            "running": bot_engine.state.running,
            "paused": bot_engine.state.paused,
            "mode": bot_engine.mode.name,
            "pairs": bot_engine.pairs,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("Failed to get bot status")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to get bot status",
                    "details": {"exception": str(e)}
                }
            }
        )


@api_router.get("/orders", dependencies=[Depends(verify_api_key)] if API_KEY else [])
async def get_orders(db: Session = Depends(get_db)):
    """Get recent orders/trades."""
    try:
        # Query recent orders from database
        orders_query = db.query(Order).order_by(Order.timestamp.desc()).limit(10).all()

        orders = []
        for order in orders_query:
            orders.append({
                "id": order.id,
                "symbol": order.symbol,
                "timestamp": order.timestamp.isoformat() if order.timestamp else None,
                "side": order.side,
                "quantity": order.quantity,
                "price": order.price,
                "pnl": order.pnl,
                "equity": order.equity,
                "cumulative_return": order.cumulative_return
            })

        return {"orders": orders}
    except Exception as e:
        logger.exception("Failed to get orders")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to get orders",
                    "details": {"exception": str(e)}
                }
            }
        )


@api_router.get("/signals", dependencies=[Depends(verify_api_key)] if API_KEY else [])
async def get_signals(db: Session = Depends(get_db)):
    """Get recent trading signals."""
    try:
        # Query recent signals from database
        signals_query = db.query(Signal).order_by(Signal.timestamp.desc()).limit(10).all()

        signals = []
        for signal in signals_query:
            signals.append({
                "id": signal.id,
                "symbol": signal.symbol,
                "timestamp": signal.timestamp.isoformat() if signal.timestamp else None,
                "confidence": signal.confidence,
                "signal_type": signal.signal_type,
                "strategy": signal.strategy
            })

        return {"signals": signals}
    except Exception as e:
        logger.exception("Failed to get signals")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to get signals",
                    "details": {"exception": str(e)}
                }
            }
        )


@api_router.get("/equity", dependencies=[Depends(verify_api_key)] if API_KEY else [])
async def get_equity(db: Session = Depends(get_db)):
    """Get equity curve data."""
    try:
        # Query equity data from database
        equity_query = db.query(Equity).order_by(Equity.timestamp.asc()).all()

        equity_data = []
        for equity_point in equity_query:
            equity_data.append({
                "timestamp": equity_point.timestamp.isoformat() if equity_point.timestamp else None,
                "balance": equity_point.balance,
                "equity": equity_point.equity,
                "cumulative_return": equity_point.cumulative_return
            })

        return {"equity_curve": equity_data}
    except Exception as e:
        logger.exception("Failed to get equity data")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to get equity data",
                    "details": {"exception": str(e)}
                }
            }
        )


@api_router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    if not bot_engine:
        return {"status": "unhealthy", "detail": "Bot engine not available"}

    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "bot_running": bot_engine.state.running
        }
    except Exception as e:
        logger.exception("Health check failed")
        return {"status": "unhealthy", "detail": str(e)}


@api_router.post("/pause", dependencies=[Depends(verify_api_key)] if API_KEY else [])
async def pause_bot():
    """Pause the bot."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": 503,
                    "message": "Bot engine not available"
                }
            }
        )

    try:
        bot_engine.state.paused = True
        logger.info("Bot paused via API")
        return {"message": "Bot paused successfully"}
    except Exception as e:
        logger.exception("Failed to pause bot")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to pause bot",
                    "details": {"exception": str(e)}
                }
            }
        )


@api_router.post("/resume", dependencies=[Depends(verify_api_key)] if API_KEY else [])
async def resume_bot():
    """Resume the bot."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": 503,
                    "message": "Bot engine not available"
                }
            }
        )

    try:
        bot_engine.state.paused = False
        logger.info("Bot resumed via API")
        return {"message": "Bot resumed successfully"}
    except Exception as e:
        logger.exception("Failed to resume bot")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to resume bot",
                    "details": {"exception": str(e)}
                }
            }
        )


@api_router.get("/performance", dependencies=[Depends(verify_api_key)] if API_KEY else [])
async def get_performance():
    """Get performance metrics."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": 503,
                    "message": "Bot engine not available"
                }
            }
        )

    try:
        return {
            "total_pnl": bot_engine.performance_stats.get("total_pnl", 0.0),
            "win_rate": bot_engine.performance_stats.get("win_rate", 0.0),
            "wins": bot_engine.performance_stats.get("wins", 0),
            "losses": bot_engine.performance_stats.get("losses", 0),
            "sharpe_ratio": bot_engine.performance_stats.get("sharpe_ratio", 0.0),
            "max_drawdown": bot_engine.performance_stats.get("max_drawdown", 0.0)
        }
    except Exception as e:
        logger.exception("Failed to get performance metrics")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": 500,
                    "message": "Failed to get performance metrics",
                    "details": {"exception": str(e)}
                }
            }
        )


@api_router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard")
async def dashboard(request: Request):
    """Serve the dashboard HTML page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Include the API router
app.include_router(api_router)
