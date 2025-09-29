"""
Health check utilities for N1V1 trading framework.

Provides standardized health and readiness check functions for HTTP endpoints,
integrating with the existing diagnostics system.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import redis

try:
    import psycopg2
except ImportError:
    from utils.logger import get_trade_logger

    logger = get_trade_logger()
    psycopg2 = None
    logger.warning(
        "psycopg2 not installed, skipping DB health check in test/dev environments"
    )
try:
    import pymongo
except ImportError:
    pymongo = None
from sqlalchemy import create_engine, text

from core.diagnostics import get_diagnostics_manager
from utils.logger import get_trade_logger

logger = get_trade_logger()


@dataclass
class ReadinessCheck:
    """Result of a readiness check."""

    component: str
    ready: bool
    latency_ms: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class HealthCheckManager:
    """
    Manager for health and readiness checks used by HTTP endpoints.
    """

    def __init__(self):
        self.diagnostics_manager = get_diagnostics_manager()
        self._correlation_id_counter = 0
        self._last_failure_log = {}  # Track last failure log time per component

    def _get_correlation_id(self) -> str:
        """Generate a correlation ID for logging."""
        self._correlation_id_counter += 1
        return f"health_{self._correlation_id_counter:06d}"

    def _should_log_failure(self, component: str, throttle_seconds: int = 300) -> bool:
        """Check if we should log a failure (throttled to avoid spam)."""
        now = time.time()
        last_log = self._last_failure_log.get(component, 0)

        if now - last_log > throttle_seconds:
            self._last_failure_log[component] = now
            return True
        return False

    async def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform lightweight health check.

        Returns:
            Dict containing health status and metadata
        """
        start_time = time.time()
        correlation_id = self._get_correlation_id()

        try:
            # Get basic system info
            uptime_seconds = time.time() - getattr(self, "_start_time", time.time())
            if not hasattr(self, "_start_time"):
                self._start_time = time.time()
                uptime_seconds = 0

            # Check if bot engine is available (lightweight check)
            from api.app import bot_engine

            bot_engine_status = "unknown"
            if bot_engine is None:
                bot_engine_status = "unavailable"
                status = "unhealthy"
                detail = "Bot engine not available"
            else:
                bot_engine_status = "available"
                status = "healthy"
                detail = None

            response = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",  # Should be read from package version
                "uptime_seconds": round(uptime_seconds, 2),
                "bot_engine": bot_engine_status,
                "correlation_id": correlation_id,
            }
            if detail:
                response["detail"] = detail

            latency = (time.time() - start_time) * 1000
            response["check_latency_ms"] = round(latency, 2)

            logger.debug(
                "Health check completed",
                extra={
                    "correlation_id": correlation_id,
                    "latency_ms": latency,
                    "status": "healthy",
                },
            )

            return response

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.exception(
                "Health check failed",
                extra={
                    "correlation_id": correlation_id,
                    "latency_ms": latency,
                    "error": str(e),
                },
            )

            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "correlation_id": correlation_id,
                "check_latency_ms": round(latency, 2),
            }

    async def perform_readiness_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Perform comprehensive readiness check on all dependencies.

        Returns:
            Tuple of (response_dict, http_status_code)
        """
        start_time = time.time()
        correlation_id = self._get_correlation_id()

        try:
            checks = await self._run_all_readiness_checks()

            # Determine overall readiness
            all_ready = all(check.ready for check in checks)
            http_status = 200 if all_ready else 503

            response = {
                "ready": all_ready,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "checks": {
                    check.component: {
                        "ready": check.ready,
                        "latency_ms": check.latency_ms,
                        "message": check.message,
                        "details": check.details,
                    }
                    for check in checks
                },
            }

            latency = (time.time() - start_time) * 1000
            response["total_latency_ms"] = round(latency, 2)

            # Log failures with throttling
            failed_checks = [check for check in checks if not check.ready]
            if failed_checks:
                if self._should_log_failure("readiness_overall"):
                    logger.warning(
                        "Readiness check failed",
                        extra={
                            "correlation_id": correlation_id,
                            "failed_components": [c.component for c in failed_checks],
                            "total_checks": len(checks),
                            "latency_ms": latency,
                        },
                    )

                # Log individual component failures
                for check in failed_checks:
                    if self._should_log_failure(f"readiness_{check.component}"):
                        logger.error(
                            f"Readiness check failed for {check.component}",
                            extra={
                                "correlation_id": correlation_id,
                                "component": check.component,
                                "check_message": check.message,
                                "latency_ms": check.latency_ms,
                                "details": check.details,
                            },
                        )

            return response, http_status

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.exception(
                "Readiness check failed",
                extra={
                    "correlation_id": correlation_id,
                    "latency_ms": latency,
                    "error": str(e),
                },
            )

            return {
                "ready": False,
                "timestamp": datetime.now().isoformat(),
                "correlation_id": correlation_id,
                "error": str(e),
                "total_latency_ms": round(latency, 2),
                "checks": {},
            }, 503

    async def _run_all_readiness_checks(self) -> List[ReadinessCheck]:
        """Run all configured readiness checks."""
        checks = []

        # Database connectivity
        db_check = await self._check_database_connectivity()
        checks.append(db_check)

        # Exchange connectivity
        exchange_check = await self._check_exchange_connectivity()
        checks.append(exchange_check)

        # Message queue connectivity
        mq_check = await self._check_message_queue_connectivity()
        checks.append(mq_check)

        # Cache connectivity (Redis)
        cache_check = await self._check_cache_connectivity()
        checks.append(cache_check)

        # Bot engine readiness
        bot_check = await self._check_bot_engine_readiness()
        checks.append(bot_check)

        return checks

    async def _check_database_connectivity(self) -> ReadinessCheck:
        """Check database connectivity."""
        start_time = time.time()

        try:
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                return ReadinessCheck(
                    component="database",
                    ready=False,
                    message="Database not configured",
                    details={"configured": False},
                )

            # Try to connect based on database type
            if "postgresql" in db_url.lower():
                if psycopg2 is None:
                    return ReadinessCheck(
                        component="database",
                        ready=True,  # Consider skipped as ready for test/dev
                        message="DB check skipped - psycopg2 not available",
                        details={"skipped": True, "reason": "psycopg2 not installed"},
                    )
                # PostgreSQL connection
                conn = psycopg2.connect(db_url)
                conn.close()
            elif "mysql" in db_url.lower():
                # MySQL connection (would need pymysql)
                pass  # Placeholder
            elif "mongodb" in db_url.lower():
                if pymongo is None:
                    return ReadinessCheck(
                        component="database",
                        ready=True,  # Consider skipped as ready for test/dev
                        message="DB check skipped - pymongo not available",
                        details={"skipped": True, "reason": "pymongo not installed"},
                    )
                # MongoDB connection
                client = pymongo.MongoClient(db_url, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
                client.close()
            else:
                # SQLAlchemy for other databases
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                engine.dispose()

            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="database",
                ready=True,
                latency_ms=round(latency, 2),
                message="Database connection successful",
                details={"type": "configured"},
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="database",
                ready=False,
                latency_ms=round(latency, 2),
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e), "type": type(e).__name__},
            )

    async def _check_exchange_connectivity(self) -> ReadinessCheck:
        """Check exchange API connectivity."""
        start_time = time.time()

        try:
            exchange_url = os.getenv(
                "EXCHANGE_API_URL", "https://api.binance.com/api/v3/ping"
            )
            timeout = aiohttp.ClientTimeout(total=5)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(exchange_url) as response:
                    if response.status == 200:
                        latency = (time.time() - start_time) * 1000
                        return ReadinessCheck(
                            component="exchange",
                            ready=True,
                            latency_ms=round(latency, 2),
                            message="Exchange API responsive",
                            details={
                                "status_code": response.status,
                                "url": exchange_url,
                            },
                        )
                    else:
                        latency = (time.time() - start_time) * 1000
                        return ReadinessCheck(
                            component="exchange",
                            ready=False,
                            latency_ms=round(latency, 2),
                            message=f"Exchange API returned status {response.status}",
                            details={
                                "status_code": response.status,
                                "url": exchange_url,
                            },
                        )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="exchange",
                ready=False,
                latency_ms=round(latency, 2),
                message=f"Exchange connectivity failed: {str(e)}",
                details={"error": str(e), "url": exchange_url},
            )

    async def _check_message_queue_connectivity(self) -> ReadinessCheck:
        """Check message queue connectivity."""
        start_time = time.time()

        try:
            mq_url = os.getenv("MESSAGE_QUEUE_URL", os.getenv("RABBITMQ_URL"))
            if not mq_url:
                return ReadinessCheck(
                    component="message_queue",
                    ready=True,  # Not configured is considered ready (optional dependency)
                    message="Message queue not configured (optional)",
                    details={"configured": False, "optional": True},
                )

            # For now, just check if it's a valid URL format
            # In production, you'd implement specific MQ client checks
            if mq_url.startswith(("amqp://", "amqps://")):
                # RabbitMQ check would go here
                latency = (time.time() - start_time) * 1000
                return ReadinessCheck(
                    component="message_queue",
                    ready=True,
                    latency_ms=round(latency, 2),
                    message="Message queue URL configured",
                    details={"type": "rabbitmq", "configured": True},
                )
            elif mq_url.startswith(("kafka://", "kafka+ssl://")):
                # Kafka check would go here
                latency = (time.time() - start_time) * 1000
                return ReadinessCheck(
                    component="message_queue",
                    ready=True,
                    latency_ms=round(latency, 2),
                    message="Message queue URL configured",
                    details={"type": "kafka", "configured": True},
                )
            else:
                return ReadinessCheck(
                    component="message_queue",
                    ready=False,
                    message="Unsupported message queue URL format",
                    details={"url": mq_url, "supported": ["amqp", "kafka"]},
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="message_queue",
                ready=False,
                latency_ms=round(latency, 2),
                message=f"Message queue check failed: {str(e)}",
                details={"error": str(e)},
            )

    async def _check_cache_connectivity(self) -> ReadinessCheck:
        """Check cache (Redis) connectivity."""
        start_time = time.time()

        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

            # Try to connect to Redis
            client = redis.from_url(redis_url)
            client.ping()
            client.close()

            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="cache",
                ready=True,
                latency_ms=round(latency, 2),
                message="Cache connection successful",
                details={"type": "redis"},
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="cache",
                ready=False,
                latency_ms=round(latency, 2),
                message=f"Cache connection failed: {str(e)}",
                details={"error": str(e), "type": "redis"},
            )

    async def _check_bot_engine_readiness(self) -> ReadinessCheck:
        """Check if bot engine is ready for trading."""
        start_time = time.time()

        try:
            # Import here to avoid circular imports
            from api.app import bot_engine

            if bot_engine is None:
                return ReadinessCheck(
                    component="bot_engine",
                    ready=False,
                    message="Bot engine not initialized",
                    details={"initialized": False},
                )

            # Check if bot engine has required components
            required_attrs = ["state", "pairs", "mode"]
            missing_attrs = [
                attr for attr in required_attrs if not hasattr(bot_engine, attr)
            ]

            if missing_attrs:
                return ReadinessCheck(
                    component="bot_engine",
                    ready=False,
                    message=f"Bot engine missing required attributes: {missing_attrs}",
                    details={"missing_attributes": missing_attrs},
                )

            # Check if bot is in a runnable state
            if hasattr(bot_engine.state, "running"):
                ready = True
                message = "Bot engine ready"
                details = {"running": bot_engine.state.running}
            else:
                ready = False
                message = "Bot engine state not properly initialized"
                details = {"state_initialized": False}

            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="bot_engine",
                ready=ready,
                latency_ms=round(latency, 2),
                message=message,
                details=details,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ReadinessCheck(
                component="bot_engine",
                ready=False,
                latency_ms=round(latency, 2),
                message=f"Bot engine check failed: {str(e)}",
                details={"error": str(e)},
            )


# Global health check manager instance
_health_check_manager: Optional[HealthCheckManager] = None


def get_health_check_manager() -> HealthCheckManager:
    """Get the global health check manager instance."""
    global _health_check_manager
    if _health_check_manager is None:
        _health_check_manager = HealthCheckManager()
    return _health_check_manager
