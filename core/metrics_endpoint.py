"""
Prometheus Metrics Endpoint for N1V1 Trading Framework

This module provides a dedicated HTTP endpoint (/metrics) that serves
metrics in Prometheus exposition format. The endpoint is optimized for
high-performance scraping with minimal latency impact.

Features:
- Fast, async HTTP endpoint using aiohttp
- Prometheus exposition format compliance
- Configurable authentication and TLS
- Health checks and endpoint monitoring
- Integration with existing metrics collector
"""

import asyncio
import time
from typing import Dict, Any, Optional
import aiohttp
from aiohttp import web
import ssl
import json

from core.metrics_collector import get_metrics_collector, MetricsCollector
from .logging_utils import get_structured_logger
from .logging_utils import LogSensitivity

logger = get_structured_logger("metrics_endpoint", LogSensitivity.INFO)


class MetricsEndpoint:
    """
    Prometheus metrics HTTP endpoint server.

    Provides a high-performance HTTP endpoint that serves metrics in
    Prometheus exposition format with minimal latency impact.

    The constructor accepts either a configuration dictionary or a MetricsCollector instance.
    If a MetricsCollector is provided, it will be used directly; otherwise, the global collector is used.
    """

    def __init__(self, config_or_collector):
        if isinstance(config_or_collector, MetricsCollector):
            self.metrics_collector = config_or_collector
            self.config = getattr(self.metrics_collector, 'config', {})  # Try to get config from collector
        elif isinstance(config_or_collector, dict):
            self.config = config_or_collector
            self.metrics_collector = get_metrics_collector()
        else:
            raise TypeError("MetricsEndpoint expects a dict config or MetricsCollector instance")

        # Server configuration
        self.host = self.config.get('host', '0.0.0.0')
        self.port = self.config.get('port', 9090)
        self.path = self.config.get('path', '/metrics')

        # Security configuration - secure defaults
        self.enable_auth = self.config.get('enable_auth', True)  # secure default
        self.auth_token = self.config.get('auth_token', '')
        if not self.auth_token and self.enable_auth:
            logger.warning("Authentication enabled but no auth_token provided - metrics endpoint will reject all requests")

        # Apply secure defaults
        self.enable_tls = self.config.get("enable_tls", True)
        self.cert_file = self.config.get("cert_file")
        self.key_file = self.config.get("key_file")

        # Validation: downgrade TLS if explicitly enabled but incomplete
        if "enable_tls" in self.config and self.config["enable_tls"] and not (self.cert_file and self.key_file):
            logger.warning(
                "TLS enabled but cert_file/key_file not provided - falling back to HTTP"
            )
            self.enable_tls = False

        # Performance configuration
        self.max_concurrent_requests = self.config.get('max_concurrent_requests', 10)
        self.request_timeout = self.config.get('request_timeout', 5.0)

        # Server state
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._running = False

        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.avg_response_time = 0.0

        logger.info(f"MetricsEndpoint initialized on {self.host}:{self.port}{self.path}", component="metrics_endpoint", operation="initialize", host=self.host, port=self.port, path=self.path)

    def create_app(self) -> web.Application:
        """Create aiohttp application for the metrics endpoint."""
        app = web.Application()

        # Add middleware
        app.middlewares.append(self._request_middleware)
        if self.enable_auth:
            app.middlewares.append(self._auth_middleware)

        # Add routes
        app.router.add_get(self.path, self._metrics_handler)
        app.router.add_get('/health', self._health_handler)
        app.router.add_get('/', self._root_handler)

        return app

    async def start(self) -> None:
        """Start the metrics endpoint server."""
        if self._running:
            return

        ssl_context = None
        try:
            # Create aiohttp application
            self.app = web.Application()

            # Add middleware
            self.app.middlewares.append(self._request_middleware)
            if self.enable_auth:
                self.app.middlewares.append(self._auth_middleware)

            # Add routes
            self.app.router.add_get(self.path, self._metrics_handler)
            self.app.router.add_get('/health', self._health_handler)
            self.app.router.add_get('/', self._root_handler)

            # Configure SSL if enabled
            if self.enable_tls and self.cert_file and self.key_file:
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(self.cert_file, self.key_file)
                logger.info("SSL/TLS configured for metrics endpoint", component="metrics_endpoint", operation="start")

            # Create runner and site
            self.runner = web.AppRunner(
                self.app,
                access_log=None,  # Disable access logging for performance
                keepalive_timeout=30.0
            )
            await self.runner.setup()

            self.site = web.TCPSite(
                self.runner,
                self.host,
                self.port,
                ssl_context=ssl_context
            )
            await self.site.start()

            self._running = True
            logger.info(f"✅ Metrics endpoint started on {self.host}:{self.port}{self.path}", component="metrics_endpoint", operation="start", host=self.host, port=self.port, path=self.path)

        except FileNotFoundError as e:
            logger.error(f"SSL certificate/key file not found: {e}", component="metrics_endpoint", operation="start", error=str(e))
            raise
        except ssl.SSLError as e:
            logger.error(f"SSL configuration error: {e}", component="metrics_endpoint", operation="start", error=str(e))
            raise
        except OSError as e:
            logger.error(f"Network binding error: {e}", component="metrics_endpoint", operation="start", error=str(e))
            raise
        except Exception as e:
            logger.exception(f"Failed to start metrics endpoint: {e}", component="metrics_endpoint", operation="start", error=str(e))
            # Ensure cleanup on failure
            await self._cleanup_on_failure()
            raise

    async def stop(self) -> None:
        """Stop the metrics endpoint server."""
        if not self._running:
            return

        try:
            if self.site:
                await self.site.stop()

            if self.runner:
                await self.runner.cleanup()

            self._running = False
            logger.info("✅ Metrics endpoint stopped", component="metrics_endpoint", operation="stop")

        except Exception as e:
            logger.exception(f"Error stopping metrics endpoint: {e}", component="metrics_endpoint", operation="stop", error=str(e))
            # Ensure cleanup even on error
            await self._cleanup_on_failure()

    async def _cleanup_on_failure(self) -> None:
        """Cleanup resources when startup or operation fails."""
        try:
            if self.site:
                try:
                    await self.site.stop()
                except Exception:
                    pass  # Ignore errors during cleanup

            if self.runner:
                try:
                    await self.runner.cleanup()
                except Exception:
                    pass  # Ignore errors during cleanup

            # Reset state
            self.site = None
            self.runner = None
            self.app = None
            self._running = False

            logger.debug("Metrics endpoint cleanup completed after failure", component="metrics_endpoint", operation="cleanup")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", component="metrics_endpoint", operation="cleanup", error=str(e))

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    @web.middleware
    async def _request_middleware(self, request: web.Request, handler):
        """Request middleware for performance tracking."""
        start_time = time.time()

        try:
            response = await handler(request)

            # Track performance
            response_time = time.time() - start_time
            self.request_count += 1

            # Update average response time
            if self.avg_response_time == 0:
                self.avg_response_time = response_time
            else:
                self.avg_response_time = (self.avg_response_time + response_time) / 2

            # Add performance headers
            response.headers['X-Response-Time'] = f"{response_time:.3f}s"
            response.headers['X-Request-Count'] = str(self.request_count)

            return response

        except Exception as e:
            self.error_count += 1
            logger.exception(f"Request error: {e}")
            raise

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        """Authentication middleware."""
        if not self.enable_auth:
            return await handler(request)

        # Check for authorization header
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            raise web.HTTPUnauthorized(text="Missing or invalid authorization header")

        token = auth_header[7:]  # Remove 'Bearer ' prefix

        if token != self.auth_token:
            raise web.HTTPUnauthorized(text="Invalid authentication token")

        return await handler(request)

    async def _metrics_handler(self, request: web.Request) -> web.Response:
        """Handle Prometheus metrics requests."""
        try:
            # Get metrics output
            metrics_output = self.metrics_collector.get_prometheus_output()

            # Add endpoint-specific metrics
            endpoint_metrics = self._get_endpoint_metrics()
            full_output = metrics_output + "\n" + endpoint_metrics

            return web.Response(
                text=full_output,
                content_type='text/plain; version=0.0.4',
                charset='utf-8',
                headers={
                    'Content-Encoding': 'identity',
                    'Cache-Control': 'no-cache, no-store, must-revalidate'
                }
            )

        except Exception as e:
            logger.exception(f"Error generating metrics output: {e}")
            self.error_count += 1
            raise web.HTTPInternalServerError(text="Error generating metrics")

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests with efficient JSON serialization."""
        import orjson

        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - getattr(self, '_start_time', time.time()),
            "metrics_collector": "healthy" if self.metrics_collector else "unhealthy",
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time": self.avg_response_time
        }

        # Use orjson for faster JSON serialization if available, fallback to json
        try:
            import orjson
            json_data = orjson.dumps(health_status).decode('utf-8')
        except ImportError:
            import json
            json_data = json.dumps(health_status)

        return web.Response(
            text=json_data,
            content_type='application/json',
            charset='utf-8'
        )

    async def _root_handler(self, request: web.Request) -> web.Response:
        """Handle root path requests with efficient JSON serialization."""
        info = {
            "service": "N1V1 Trading Framework Metrics Endpoint",
            "version": "1.0.0",
            "endpoints": {
                "/metrics": "Prometheus metrics exposition format",
                "/health": "Health check endpoint",
                "/": "Service information"
            },
            "status": "running" if self._running else "stopped"
        }

        # Use orjson for faster JSON serialization if available, fallback to json
        try:
            import orjson
            json_data = orjson.dumps(info).decode('utf-8')
        except ImportError:
            import json
            json_data = json.dumps(info)

        return web.Response(
            text=json_data,
            content_type='application/json',
            charset='utf-8'
        )

    def _get_endpoint_metrics(self) -> str:
        """Generate endpoint-specific metrics."""
        metrics_lines = []

        # Endpoint request metrics
        metrics_lines.append("# HELP metrics_endpoint_requests_total Total number of requests to metrics endpoint")
        metrics_lines.append("# TYPE metrics_endpoint_requests_total counter")
        metrics_lines.append(f"metrics_endpoint_requests_total {self.request_count}")

        # Endpoint error metrics
        metrics_lines.append("# HELP metrics_endpoint_errors_total Total number of errors in metrics endpoint")
        metrics_lines.append("# TYPE metrics_endpoint_errors_total counter")
        metrics_lines.append(f"metrics_endpoint_errors_total {self.error_count}")

        # Endpoint performance metrics
        metrics_lines.append("# HELP metrics_endpoint_response_time_seconds Average response time for metrics endpoint")
        metrics_lines.append("# TYPE metrics_endpoint_response_time_seconds gauge")
        metrics_lines.append(f"metrics_endpoint_response_time_seconds {self.avg_response_time}")

        # Endpoint status
        metrics_lines.append("# HELP metrics_endpoint_status Status of metrics endpoint (1=running, 0=stopped)")
        metrics_lines.append("# TYPE metrics_endpoint_status gauge")
        metrics_lines.append(f"metrics_endpoint_status {1 if self._running else 0}")

        return "\n".join(metrics_lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get endpoint statistics."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time": self.avg_response_time,
            "auth_enabled": self.enable_auth,
            "tls_enabled": self.enable_tls
        }


# Global endpoint instance
_metrics_endpoint: Optional[MetricsEndpoint] = None
_endpoint_cleanup_registered = False


def get_metrics_endpoint() -> Optional[MetricsEndpoint]:
    """Get the global metrics endpoint instance."""
    return _metrics_endpoint


def create_metrics_endpoint(config: Optional[Dict[str, Any]] = None) -> MetricsEndpoint:
    """Create a new metrics endpoint instance."""
    global _metrics_endpoint, _endpoint_cleanup_registered
    _metrics_endpoint = MetricsEndpoint(config or {})

    # Register cleanup handler for application exit
    if not _endpoint_cleanup_registered:
        import atexit
        atexit.register(_cleanup_endpoint_on_exit)
        _endpoint_cleanup_registered = True
        logger.info("Metrics endpoint cleanup handler registered with atexit", component="metrics_endpoint", operation="initialize")

    return _metrics_endpoint


def _cleanup_endpoint_on_exit() -> None:
    """Cleanup function called on application exit."""
    logger.info("Performing metrics endpoint cleanup on application exit", component="metrics_endpoint", operation="cleanup")
    try:
        # Use asyncio.run() for synchronous cleanup
        asyncio.run(stop_metrics_endpoint())
        logger.info("Metrics endpoint cleanup completed successfully", component="metrics_endpoint", operation="cleanup")
    except Exception as e:
        logger.error(f"Error during metrics endpoint cleanup on exit: {str(e)}", component="metrics_endpoint", operation="cleanup", error=str(e))


# Example configuration
DEFAULT_ENDPOINT_CONFIG = {
    "host": "0.0.0.0",
    "port": 9090,
    "path": "/metrics",
    "enable_auth": False,
    "auth_token": "",
    "enable_tls": False,
    "cert_file": "",
    "key_file": "",
    "max_concurrent_requests": 10,
    "request_timeout": 5.0
}


async def start_metrics_endpoint(config: Optional[Dict[str, Any]] = None) -> MetricsEndpoint:
    """Start the metrics endpoint with the given configuration."""
    endpoint = create_metrics_endpoint(config or DEFAULT_ENDPOINT_CONFIG)
    await endpoint.start()
    return endpoint


async def stop_metrics_endpoint() -> None:
    """Stop the global metrics endpoint."""
    global _metrics_endpoint
    if _metrics_endpoint:
        await _metrics_endpoint.stop()
        _metrics_endpoint = None


# Integration with existing framework
async def integrate_with_bot_engine(bot_engine) -> None:
    """Integrate metrics endpoint with BotEngine."""
    # Add trading metrics collectors
    from core.metrics_collector import (
        collect_trading_metrics,
        collect_risk_metrics,
        collect_strategy_metrics,
        collect_exchange_metrics,
        collect_binary_model_metrics
    )

    collector = get_metrics_collector()
    collector.add_custom_collector(collect_trading_metrics)
    collector.add_custom_collector(collect_risk_metrics)
    collector.add_custom_collector(collect_strategy_metrics)
    collector.add_custom_collector(collect_exchange_metrics)
    collector.add_custom_collector(collect_binary_model_metrics)

    # Start metrics collection
    await collector.start()

    logger.info("✅ Metrics collection integrated with BotEngine")


# Example usage
if __name__ == "__main__":
    async def main():
        # Create and start metrics endpoint
        config = {
            "host": "localhost",
            "port": 9090,
            "enable_auth": False
        }

        endpoint = await start_metrics_endpoint(config)

        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await stop_metrics_endpoint()

    asyncio.run(main())
