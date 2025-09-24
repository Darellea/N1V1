"""
FastAPI application for the crypto trading bot.
Provides REST endpoints for monitoring and controlling the bot.
"""

from fastapi import FastAPI, HTTPException, Request, Depends, Response, APIRouter
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from typing import Dict, List, Any, Optional
import logging
import os
from datetime import datetime
from sqlalchemy.orm import Session
from .models import Order, Signal, Equity, get_db
from .schemas import ErrorResponse
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
import redis
from .middleware import CustomExceptionMiddleware, RateLimitExceptionMiddleware, RateLimitException



# Global reference to bot engine (will be set when app starts)
bot_engine = None

# API Key authentication - will be checked dynamically
security = HTTPBearer(auto_error=False)

logger = logging.getLogger(__name__)

# Rate limiting configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

def get_remote_address_exempt(request):
    """Get remote address, but exempt certain endpoints from rate limiting."""
    if request.url.path in ["/", "/dashboard"]:
        return None  # Exempt from rate limiting
    try:
        return get_remote_address(request)
    except AttributeError:
        return "127.0.0.1"

def on_breach(request):
    """Raise RateLimitExceeded exception to be handled by our custom handler."""
    raise RateLimitExceeded()

# Try to connect to Redis, fallback to in-memory if not available
limiter = None
if os.environ.get("TESTING") == "1":
    # Use normal limits for tests but with isolated storage
    limiter = Limiter(key_func=get_remote_address_exempt, default_limits=["60/minute"], headers_enabled=True)
else:
    try:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()  # Test connection
        limiter = Limiter(key_func=get_remote_address_exempt, default_limits=["60/minute"], storage_uri=REDIS_URL, headers_enabled=True)
        logger.info("Rate limiting configured with Redis")
    except (redis.ConnectionError, redis.TimeoutError):
        logger.warning("Redis not available, falling back to in-memory rate limiting")
        limiter = Limiter(key_func=get_remote_address_exempt, default_limits=["60/minute"], headers_enabled=True)  # In-memory fallback

app = FastAPI(
    title="Crypto Trading Bot API",
    description="REST API for monitoring and controlling the crypto trading bot",
    version="1.0.0"
)

# Add CORS middleware first
if os.environ.get("TESTING") == "1":
    # In test mode, configure CORS to echo origins for proper testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # In production, allow broader origins as fallback
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Fallback for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Configure SlowAPI
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(RateLimitExceptionMiddleware)

# Add custom exception middleware last
app.add_middleware(CustomExceptionMiddleware)

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


def format_error(code: int, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Centralized error formatting function."""
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details
        }
    }


@app.exception_handler(RateLimitException)
async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
    """Custom rate limit exceeded handler with standardized JSON response."""
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "rate_limit_exceeded",
                "message": "Rate limit exceeded",
                "details": {
                    "limit": 60,
                    "window": "1 minute",
                    "endpoint": str(request.url.path)
                }
            }
        },
        headers=exc.headers
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit exceeded handler with standardized JSON response."""
    # Parse the original SlowAPI error message
    # Format: "Rate limit exceeded: X per Y minute(s)"
    original_message = str(exc)
    limit = 60  # Default
    window = "1 minute"  # Default

    # Try to extract limit and window from the message
    if "Rate limit exceeded:" in original_message:
        try:
            # Extract "X per Y minute" part
            parts = original_message.split(": ")[1].split(" per ")
            if len(parts) == 2:
                limit_part = parts[0]
                window_part = parts[1]
                limit = int(limit_part) if limit_part.isdigit() else 60
                window = window_part
        except (IndexError, ValueError):
            # Use defaults if parsing fails
            pass

    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "rate_limit_exceeded",
                "message": original_message,
                "details": {
                    "limit": limit,
                    "window": window,
                    "endpoint": str(request.url.path)
                }
            }
        },
        headers=exc.headers
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler to remove 'detail' wrapper."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        # Already formatted error
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    else:
        # Plain string detail, format it
        return JSONResponse(
            status_code=exc.status_code,
            content=format_error(exc.status_code, str(exc.detail))
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.exception("Unhandled exception in %s %s", request.method, request.url.path, exc_info=exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "An unexpected error occurred",
                "details": {
                    "path": str(request.url.path),
                    "method": request.method
                }
            }
        }
    )


async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key if authentication is enabled."""
    current_api_key = os.getenv("API_KEY")
    if current_api_key and credentials:
        if credentials.credentials != current_api_key:
            raise HTTPException(
                status_code=401,
                detail=format_error(401, "Invalid API key")
            )
    elif current_api_key and not credentials:
        raise HTTPException(
            status_code=401,
            detail=format_error(401, "API key required")
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


@api_router.get("/status", dependencies=[Depends(verify_api_key)])
async def get_status(request: Request):
    """Get bot status."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail="Bot engine not available"
        )

    return {
        "running": bot_engine.state.running,
        "paused": bot_engine.state.paused,
        "mode": bot_engine.mode.name,
        "pairs": bot_engine.pairs,
        "timestamp": datetime.now().isoformat()
    }


@api_router.get("/orders", dependencies=[Depends(verify_api_key)])
async def get_orders(request: Request, db: Session = Depends(get_db)):
    """Get recent orders/trades."""
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


@api_router.get("/signals", dependencies=[Depends(verify_api_key)])
async def get_signals(request: Request, db: Session = Depends(get_db)):
    """Get recent trading signals."""
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


@api_router.get("/equity", dependencies=[Depends(verify_api_key)])
async def get_equity(request: Request, db: Session = Depends(get_db)):
    """Get equity curve data."""
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


@api_router.get("/health")
async def health_check(request: Request):
    """Lightweight health check endpoint."""
    from core.healthcheck import get_health_check_manager

    health_manager = get_health_check_manager()
    return await health_manager.perform_health_check()


@api_router.get("/ready")
async def readiness_check(request: Request):
    """Comprehensive readiness check endpoint."""
    from core.healthcheck import get_health_check_manager

    health_manager = get_health_check_manager()
    response, status_code = await health_manager.perform_readiness_check()

    # Return response with appropriate HTTP status
    from fastapi.responses import JSONResponse
    return JSONResponse(content=response, status_code=status_code)


@api_router.post("/pause", dependencies=[Depends(verify_api_key)])
async def pause_bot(request: Request):
    """Pause the bot."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail=format_error(503, "Bot engine not available")
        )

    bot_engine.state.paused = True
    logger.info("Bot paused via API")
    return {"message": "Bot paused successfully"}


@api_router.post("/resume", dependencies=[Depends(verify_api_key)])
async def resume_bot(request: Request):
    """Resume the bot."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail=format_error(503, "Bot engine not available")
        )

    bot_engine.state.paused = False
    logger.info("Bot resumed via API")
    return {"message": "Bot resumed successfully"}


@api_router.get("/performance", dependencies=[Depends(verify_api_key)])
async def get_performance(request: Request):
    """Get performance metrics."""
    if not bot_engine:
        raise HTTPException(
            status_code=503,
            detail=format_error(503, "Bot engine not available")
        )

    return {
        "total_pnl": bot_engine.performance_stats.get("total_pnl", 0.0),
        "win_rate": bot_engine.performance_stats.get("win_rate", 0.0),
        "wins": bot_engine.performance_stats.get("wins", 0),
        "losses": bot_engine.performance_stats.get("losses", 0),
        "sharpe_ratio": bot_engine.performance_stats.get("sharpe_ratio", 0.0),
        "max_drawdown": bot_engine.performance_stats.get("max_drawdown", 0.0)
    }


@app.get("/metrics")
async def metrics(request: Request):
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dashboard")
async def dashboard(request: Request):
    """Serve the dashboard HTML page."""
    try:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    except Exception:
        from fastapi.responses import HTMLResponse
        return HTMLResponse("<html><body>Dashboard</body></html>", status_code=200)

# Include the API router
app.include_router(api_router)
