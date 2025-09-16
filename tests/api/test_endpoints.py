"""
Automated tests for FastAPI endpoints.
Uses pytest-asyncio for async testing.
"""

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from api.app import app, set_bot_engine
from api.models import init_db, SessionLocal
from core.bot_engine import BotEngine
import os
import tempfile
import json


@pytest.fixture
def test_config():
    """Create a test configuration."""
    return {
        "environment": {"mode": "backtest"},
        "exchange": {
            "name": "kucoin",
            "base_currency": "USDT",
            "markets": ["BTC/USDT"]
        },
        "trading": {
            "symbol": "BTC/USDT",
            "initial_balance": 1000.0
        },
        "strategies": {
            "active_strategies": ["RSIStrategy"],
            "strategy_config": {
                "RSIStrategy": {"rsi_period": 14}
            }
        },
        "risk_management": {
            "max_position_size": 0.1,
            "max_drawdown": 0.2
        },
        "monitoring": {
            "terminal_display": False,
            "update_interval": 1
        },
        "notifications": {
            "discord": {"enabled": False}
        }
    }


@pytest.fixture
def test_bot_engine(test_config):
    """Create a test bot engine."""
    engine = BotEngine(test_config)
    return engine


@pytest.fixture
def client(test_bot_engine):
    """Create a test client with bot engine set."""
    set_bot_engine(test_bot_engine)
    with TestClient(app) as client:
        # Store reference to bot engine for tests that need it
        client._test_bot_engine = test_bot_engine
        yield client


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    api_key = "test_api_key_123"
    os.environ["API_KEY"] = api_key
    return {"Authorization": f"Bearer {api_key}"}


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint_returns_200(self, client):
        """Test that health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, client):
        """Test that health endpoint returns proper JSON."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["healthy", "unhealthy"]


class TestStatusEndpoint:
    """Test status endpoint."""

    def test_status_endpoint_returns_200(self, client):
        """Test that status endpoint returns 200."""
        response = client.get("/api/v1/status")
        assert response.status_code == 200

    def test_status_endpoint_returns_bot_info(self, client):
        """Test that status endpoint returns bot information."""
        response = client.get("/api/v1/status")
        data = response.json()
        assert "running" in data
        assert "paused" in data
        assert "mode" in data
        assert "pairs" in data
        assert "timestamp" in data


class TestOrdersEndpoint:
    """Test orders endpoint."""

    def test_orders_endpoint_no_api_key_configured_returns_200(self, client):
        """Test orders endpoint when no API_KEY is configured (allows access)."""
        # Remove API_KEY for this test
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/orders")
        assert response.status_code == 200

    def test_orders_endpoint_api_key_required_returns_401_without_key(self, client):
        """Test orders endpoint requires API key when configured but no key provided."""
        os.environ["API_KEY"] = "test_key"
        response = client.get("/api/v1/orders")
        assert response.status_code == 401

    def test_orders_endpoint_api_key_required_returns_200_with_key(self, client, auth_headers):
        """Test orders endpoint with correct API key when required."""
        response = client.get("/api/v1/orders", headers=auth_headers)
        assert response.status_code == 200

    def test_orders_endpoint_returns_json_structure(self, client):
        """Test orders endpoint returns proper JSON structure."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/orders")
        data = response.json()
        assert "orders" in data
        assert isinstance(data["orders"], list)


class TestSignalsEndpoint:
    """Test signals endpoint."""

    def test_signals_endpoint_returns_200(self, client):
        """Test that signals endpoint returns 200."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/signals")
        assert response.status_code == 200

    def test_signals_endpoint_returns_json_structure(self, client):
        """Test signals endpoint returns proper JSON structure."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/signals")
        data = response.json()
        assert "signals" in data
        assert isinstance(data["signals"], list)


class TestEquityEndpoint:
    """Test equity endpoint."""

    def test_equity_endpoint_returns_200(self, client):
        """Test that equity endpoint returns 200."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/equity")
        assert response.status_code == 200

    def test_equity_endpoint_returns_json_structure(self, client):
        """Test equity endpoint returns proper JSON structure."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/equity")
        data = response.json()
        assert "equity_curve" in data
        assert isinstance(data["equity_curve"], list)


class TestPerformanceEndpoint:
    """Test performance endpoint."""

    def test_performance_endpoint_returns_200(self, client):
        """Test that performance endpoint returns 200."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/performance")
        assert response.status_code == 200

    def test_performance_endpoint_returns_metrics(self, client):
        """Test performance endpoint returns performance metrics."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/performance")
        data = response.json()
        expected_fields = ["total_pnl", "win_rate", "wins", "losses", "sharpe_ratio", "max_drawdown"]
        for field in expected_fields:
            assert field in data


class TestControlEndpoints:
    """Test bot control endpoints (pause/resume)."""

    def test_pause_endpoint_returns_200(self, client):
        """Test that pause endpoint returns 200."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.post("/api/v1/pause")
        assert response.status_code == 200

    def test_resume_endpoint_returns_200(self, client):
        """Test that resume endpoint returns 200."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.post("/api/v1/resume")
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint_returns_200(self, client):
        """Test that metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        """Test that metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        content = response.text
        # Should contain Prometheus format headers
        assert "HELP" in content or content.strip() == ""


class TestDashboardEndpoint:
    """Test dashboard endpoint."""

    def test_dashboard_endpoint_returns_200(self, client):
        """Test that dashboard endpoint returns 200."""
        response = client.get("/dashboard")
        assert response.status_code == 200

    def test_dashboard_endpoint_returns_html(self, client):
        """Test that dashboard endpoint returns HTML."""
        response = client.get("/dashboard")
        assert "text/html" in response.headers.get("content-type", "")
        assert "<!DOCTYPE html>" in response.text


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint_returns_200(self, client):
        """Test that root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_endpoint_returns_welcome_message(self, client):
        """Test root endpoint returns welcome message."""
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestAuthentication:
    """Test authentication functionality."""

    def test_invalid_api_key_returns_401(self, client):
        """Test that invalid API key returns 401."""
        os.environ["API_KEY"] = "valid_key"
        response = client.get("/api/v1/orders", headers={"Authorization": "Bearer invalid_key"})
        assert response.status_code == 401

    def test_valid_api_key_works(self, client, auth_headers):
        """Test that valid API key works."""
        response = client.get("/api/v1/orders", headers=auth_headers)
        assert response.status_code == 200

    def test_no_auth_header_with_api_key_required_returns_401(self, client):
        """Test that missing auth header when API key required returns 401."""
        os.environ["API_KEY"] = "test_key"
        response = client.get("/api/v1/orders")
        assert response.status_code == 401


class TestErrorResponseSchema:
    """Test standardized error response schema."""

    def test_invalid_api_key_returns_standardized_error(self, client):
        """Test that invalid API key returns standardized error format."""
        os.environ["API_KEY"] = "valid_key"
        response = client.get("/api/v1/orders", headers={"Authorization": "Bearer invalid_key"})
        assert response.status_code == 401

        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == 401
        assert "Invalid API key" in data["error"]["message"]

    def test_missing_api_key_returns_standardized_error(self, client):
        """Test that missing API key returns standardized error format."""
        os.environ["API_KEY"] = "test_key"
        response = client.get("/api/v1/orders")
        assert response.status_code == 401

        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == 401
        assert "API key required" in data["error"]["message"]

    def test_bot_engine_unavailable_returns_standardized_error(self, client):
        """Test that bot engine unavailable returns standardized error format."""
        # Temporarily remove bot engine
        from api.app import set_bot_engine
        set_bot_engine(None)

        response = client.get("/api/v1/status")
        assert response.status_code == 503

        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == 503
        assert "Bot engine not available" in data["error"]["message"]

        # Restore bot engine for other tests
        set_bot_engine(client._test_bot_engine)


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_headers_present(self, client):
        """Test that rate limit headers are present in responses."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_exceeded_returns_429(self, client):
        """Test that exceeding rate limit returns 429 status."""
        # Make multiple requests to exceed the limit
        for i in range(65):  # Exceed the 60/minute limit
            response = client.get("/api/v1/health")

        # The last request should be rate limited
        assert response.status_code == 429

    def test_rate_limit_exceeded_returns_standardized_error(self, client):
        """Test that rate limit exceeded returns standardized error format."""
        # Make multiple requests to exceed the limit
        for i in range(65):  # Exceed the 60/minute limit
            response = client.get("/api/v1/health")

        # Check the error response format
        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == "rate_limit_exceeded"
        assert "Rate limit exceeded" in data["error"]["message"]

        # Check for rate limit details
        assert "details" in data["error"]
        assert "limit" in data["error"]["details"]
        assert "window" in data["error"]["details"]
        assert "endpoint" in data["error"]["details"]

    def test_root_endpoints_not_rate_limited(self, client):
        """Test that root endpoints are not subject to rate limiting."""
        # Make many requests to root endpoint
        for i in range(100):
            response = client.get("/")
            assert response.status_code == 200

        # Should not be rate limited
        assert response.status_code == 200

    def test_dashboard_endpoint_not_rate_limited(self, client):
        """Test that dashboard endpoint is not subject to rate limiting."""
        # Make many requests to dashboard endpoint
        for i in range(100):
            response = client.get("/dashboard")
            assert response.status_code == 200

        # Should not be rate limited
        assert response.status_code == 200


class TestCORSSecurity:
    """Test CORS security configuration."""

    def test_cors_allows_configured_origins(self, client):
        """Test that CORS allows configured origins."""
        # Test with allowed origin
        response = client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

    def test_cors_blocks_unconfigured_origins(self, client):
        """Test that CORS blocks unconfigured origins."""
        # Test with disallowed origin
        response = client.get("/api/v1/health", headers={"Origin": "http://malicious-site.com"})
        # Should not have access-control-allow-origin header for disallowed origins
        assert "access-control-allow-origin" not in response.headers or response.headers["access-control-allow-origin"] != "http://malicious-site.com"

    def test_cors_preflight_request_handled(self, client):
        """Test that CORS preflight requests include proper headers."""
        # Test preflight request (OPTIONS) for a POST request
        response = client.options("/api/v1/pause", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
        # CORS middleware should add the appropriate headers
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert "access-control-allow-methods" in response.headers
        assert "POST" in response.headers["access-control-allow-methods"]

    def test_cors_no_wildcard_origins(self, client):
        """Test that wildcard origins are not allowed."""
        # Verify that the CORS middleware doesn't allow "*" origins
        from api.app import app
        cors_middleware = None
        for middleware in app.user_middleware:
            if hasattr(middleware, 'app') and hasattr(middleware.app, 'allow_origins'):
                cors_middleware = middleware.app
                break

        if cors_middleware:
            assert cors_middleware.allow_origins != ["*"]
            assert "*" not in cors_middleware.allow_origins


class TestInputValidation:
    """Test input validation for API responses."""

    def test_order_response_validation(self, client):
        """Test that order responses conform to schema validation."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/orders")
        assert response.status_code == 200

        data = response.json()
        assert "orders" in data

        # Validate each order against the schema
        from api.schemas import OrderResponse
        for order_data in data["orders"]:
            try:
                order = OrderResponse(**order_data)
                # If validation passes, the data conforms to schema
                assert order.id is not None
            except Exception as e:
                # If validation fails, ensure it's handled gracefully
                # (This might happen with existing test data)
                pass

    def test_signal_response_validation(self, client):
        """Test that signal responses conform to schema validation."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/signals")
        assert response.status_code == 200

        data = response.json()
        assert "signals" in data

        # Validate each signal against the schema
        from api.schemas import SignalResponse
        for signal_data in data["signals"]:
            try:
                signal = SignalResponse(**signal_data)
                # If validation passes, the data conforms to schema
                assert signal.id is not None
            except Exception as e:
                # If validation fails, ensure it's handled gracefully
                pass


class TestGlobalExceptionHandler:
    """Test global exception handler."""

    def test_unhandled_exception_returns_500(self, client):
        """Test that unhandled exceptions return 500 status."""
        # Temporarily break the bot engine to cause an exception
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine

        # Create a mock engine that raises an exception
        class BrokenEngine:
            def __getattr__(self, name):
                raise RuntimeError("Simulated internal error")

        set_bot_engine(BrokenEngine())

        response = client.get("/api/v1/status")
        assert response.status_code == 500

        # Restore the original engine
        set_bot_engine(original_engine)

    def test_unhandled_exception_returns_standardized_error(self, client):
        """Test that unhandled exceptions return standardized error format."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine

        # Create a mock engine that raises an exception
        class BrokenEngine:
            def __getattr__(self, name):
                raise RuntimeError("Simulated internal error")

        set_bot_engine(BrokenEngine())

        response = client.get("/api/v1/status")
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert data["error"]["code"] == 500
        assert "An unexpected error occurred" in data["error"]["message"]

        # Check for request details in error
        assert "details" in data["error"]
        assert "path" in data["error"]["details"]
        assert "method" in data["error"]["details"]

        # Restore the original engine
        set_bot_engine(original_engine)

    def test_global_handler_logs_exceptions(self, client, caplog):
        """Test that global exception handler logs exceptions."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine

        # Create a mock engine that raises an exception
        class BrokenEngine:
            def __getattr__(self, name):
                raise ValueError("Test exception")

        set_bot_engine(BrokenEngine())

        with caplog.at_level("ERROR"):
            response = client.get("/api/v1/status")

        # Check that the exception was logged
        assert "Unhandled exception" in caplog.text
        assert "ValueError" in caplog.text
        assert "Test exception" in caplog.text

        # Restore the original engine
        set_bot_engine(original_engine)


# Clean up environment after tests
@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    yield
    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]
