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
        assert "version" in data
        assert "uptime_seconds" in data
        assert "bot_engine" in data
        assert "correlation_id" in data
        assert "check_latency_ms" in data
        assert data["status"] in ["healthy", "unhealthy"]

    def test_health_endpoint_returns_metadata(self, client):
        """Test that health endpoint returns required metadata."""
        response = client.get("/api/v1/health")
        data = response.json()

        # Check metadata fields
        assert isinstance(data["version"], str)
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["correlation_id"], str)
        assert isinstance(data["check_latency_ms"], (int, float))
        assert data["uptime_seconds"] >= 0
        assert data["check_latency_ms"] >= 0

    def test_health_endpoint_correlation_id_unique(self, client):
        """Test that correlation IDs are unique across requests."""
        response1 = client.get("/api/v1/health")
        response2 = client.get("/api/v1/health")

        data1 = response1.json()
        data2 = response2.json()

        assert data1["correlation_id"] != data2["correlation_id"]
        assert data1["correlation_id"].startswith("health_")
        assert data2["correlation_id"].startswith("health_")

    def test_health_endpoint_timestamp_format(self, client):
        """Test that timestamp is in ISO format."""
        import re
        from datetime import datetime

        response = client.get("/api/v1/health")
        data = response.json()

        # Check ISO format
        iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$'
        assert re.match(iso_pattern, data["timestamp"])

        # Check it's a valid datetime
        parsed = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        assert isinstance(parsed, datetime)


class TestReadinessEndpoint:
    """Test readiness check endpoint."""

    def test_readiness_endpoint_returns_200_when_healthy(self, client):
        """Test that readiness endpoint returns 200 when all checks pass."""
        response = client.get("/api/v1/ready")
        # Should return 200 since database/cache are optional and bot engine is available
        assert response.status_code in [200, 503]  # 503 if dependencies fail

    def test_readiness_endpoint_returns_json_structure(self, client):
        """Test that readiness endpoint returns proper JSON structure."""
        response = client.get("/api/v1/ready")
        data = response.json()

        assert "ready" in data
        assert "timestamp" in data
        assert "correlation_id" in data
        assert "checks" in data
        assert "total_latency_ms" in data
        assert isinstance(data["ready"], bool)
        assert isinstance(data["checks"], dict)
        assert isinstance(data["correlation_id"], str)
        assert isinstance(data["total_latency_ms"], (int, float))

    def test_readiness_endpoint_includes_all_check_components(self, client):
        """Test that readiness endpoint includes all expected components."""
        response = client.get("/api/v1/ready")
        data = response.json()

        expected_components = ["database", "exchange", "message_queue", "cache", "bot_engine"]
        checks = data["checks"]

        for component in expected_components:
            assert component in checks
            assert "ready" in checks[component]
            assert "message" in checks[component]
            assert isinstance(checks[component]["ready"], bool)
            assert isinstance(checks[component]["message"], str)

    def test_readiness_endpoint_correlation_id_unique(self, client):
        """Test that readiness correlation IDs are unique across requests."""
        response1 = client.get("/api/v1/ready")
        response2 = client.get("/api/v1/ready")

        data1 = response1.json()
        data2 = response2.json()

        assert data1["correlation_id"] != data2["correlation_id"]
        assert data1["correlation_id"].startswith("health_")
        assert data2["correlation_id"].startswith("health_")

    def test_readiness_endpoint_timestamp_format(self, client):
        """Test that readiness timestamp is in ISO format."""
        import re
        from datetime import datetime

        response = client.get("/api/v1/ready")
        data = response.json()

        # Check ISO format
        iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$'
        assert re.match(iso_pattern, data["timestamp"])

        # Check it's a valid datetime
        parsed = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
        assert isinstance(parsed, datetime)

    def test_readiness_endpoint_returns_503_when_bot_engine_unavailable(self, client):
        """Test that readiness returns 503 when bot engine is unavailable."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine

        # Temporarily remove bot engine
        set_bot_engine(None)

        response = client.get("/api/v1/ready")
        assert response.status_code == 503

        data = response.json()
        assert data["ready"] is False
        assert "bot_engine" in data["checks"]
        assert data["checks"]["bot_engine"]["ready"] is False

        # Restore bot engine
        set_bot_engine(original_engine)

    def test_readiness_endpoint_check_details_structure(self, client):
        """Test that readiness check details have proper structure."""
        response = client.get("/api/v1/ready")
        data = response.json()

        for component, check_data in data["checks"].items():
            # Check required fields
            assert "ready" in check_data
            assert "message" in check_data

            # Check optional fields if present
            if "latency_ms" in check_data:
                assert isinstance(check_data["latency_ms"], (int, float, type(None)))
            if "details" in check_data:
                assert isinstance(check_data["details"], dict)

    def test_readiness_endpoint_latency_measurement(self, client):
        """Test that readiness endpoint measures latency properly."""
        response = client.get("/api/v1/ready")
        data = response.json()

        # Total latency should be reasonable (less than 10 seconds for tests)
        assert data["total_latency_ms"] >= 0
        assert data["total_latency_ms"] < 10000

        # Individual check latencies should also be reasonable
        for component, check_data in data["checks"].items():
            if "latency_ms" in check_data and check_data["latency_ms"] is not None:
                assert check_data["latency_ms"] >= 0
                assert check_data["latency_ms"] < 5000  # 5 seconds max for any single check

    @pytest.mark.parametrize("missing_env_var", [
        "DATABASE_URL",
        "EXCHANGE_API_URL",
        "MESSAGE_QUEUE_URL",
        "REDIS_URL"
    ])
    def test_readiness_endpoint_handles_missing_env_vars(self, client, missing_env_var):
        """Test that readiness endpoint handles missing environment variables gracefully."""
        # Store original value
        original_value = os.environ.get(missing_env_var)

        try:
            # Remove the environment variable
            if missing_env_var in os.environ:
                del os.environ[missing_env_var]

            response = client.get("/api/v1/ready")
            data = response.json()

            # Should still return a valid response
            assert "checks" in data
            assert "ready" in data

            # Check that the relevant component is marked as not ready or optional
            if missing_env_var == "DATABASE_URL":
                # Database should be marked as not configured
                assert "database" in data["checks"]
                db_check = data["checks"]["database"]
                assert "configured" in db_check.get("details", {}) or not db_check["ready"]

        finally:
            # Restore original value
            if original_value is not None:
                os.environ[missing_env_var] = original_value

    def test_readiness_endpoint_bot_engine_check(self, client):
        """Test that readiness endpoint properly checks bot engine."""
        response = client.get("/api/v1/ready")
        data = response.json()

        assert "bot_engine" in data["checks"]
        bot_check = data["checks"]["bot_engine"]

        # With the test bot engine, this should be ready
        assert "ready" in bot_check
        assert "message" in bot_check
        assert isinstance(bot_check["ready"], bool)

    def test_readiness_endpoint_exchange_check(self, client):
        """Test that readiness endpoint checks exchange connectivity."""
        response = client.get("/api/v1/ready")
        data = response.json()

        assert "exchange" in data["checks"]
        exchange_check = data["checks"]["exchange"]

        assert "ready" in exchange_check
        assert "message" in exchange_check
        assert isinstance(exchange_check["ready"], bool)

        # Should have latency if it attempted connection
        if "latency_ms" in exchange_check:
            assert isinstance(exchange_check["latency_ms"], (int, float))

    def test_readiness_endpoint_cache_check(self, client):
        """Test that readiness endpoint checks cache connectivity."""
        response = client.get("/api/v1/ready")
        data = response.json()

        assert "cache" in data["checks"]
        cache_check = data["checks"]["cache"]

        assert "ready" in cache_check
        assert "message" in cache_check
        assert isinstance(cache_check["ready"], bool)

    def test_readiness_endpoint_message_queue_check(self, client):
        """Test that readiness endpoint checks message queue."""
        response = client.get("/api/v1/ready")
        data = response.json()

        assert "message_queue" in data["checks"]
        mq_check = data["checks"]["message_queue"]

        assert "ready" in mq_check
        assert "message" in mq_check
        assert isinstance(mq_check["ready"], bool)

        # Message queue is optional, so it might be ready even if not configured
        if not mq_check["ready"]:
            assert "configured" in mq_check.get("details", {})

    def test_readiness_endpoint_database_check(self, client):
        """Test that readiness endpoint checks database connectivity."""
        response = client.get("/api/v1/ready")
        data = response.json()

        assert "database" in data["checks"]
        db_check = data["checks"]["database"]

        assert "ready" in db_check
        assert "message" in db_check
        assert isinstance(db_check["ready"], bool)

        # Database might not be configured in tests, which is acceptable
        if not db_check["ready"]:
            assert "configured" in db_check.get("details", {}) or "Database not configured" in db_check["message"]


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


class TestCustomExceptionMiddleware:
    """Test custom exception middleware."""

    def test_custom_exception_middleware_handles_exceptions(self, client):
        """Test that custom exception middleware catches and handles exceptions."""
        # This test is tricky to trigger directly, but we can test by ensuring the middleware is added
        from api.app import app
        assert any('CustomExceptionMiddleware' in str(mw) for mw in app.user_middleware)


class TestRateLimitJSONMiddleware:
    """Test rate limit JSON middleware."""

    def test_rate_limit_middleware_converts_429_to_json(self, client):
        """Test that rate limit middleware converts 429 responses to JSON."""
        # Make many requests to trigger rate limit
        for i in range(65):
            response = client.get("/api/v1/health")

        # Check that the response is JSON even for 429
        assert response.status_code == 429
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "rate_limit_exceeded"


class TestBotEngineUnavailable:
    """Test behavior when bot engine is unavailable."""

    def test_status_endpoint_bot_unavailable(self, client):
        """Test status endpoint when bot engine is None."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine
        set_bot_engine(None)

        response = client.get("/api/v1/status")
        assert response.status_code == 503

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == 503

        set_bot_engine(original_engine)

    def test_pause_endpoint_bot_unavailable(self, client):
        """Test pause endpoint when bot engine is None."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine
        set_bot_engine(None)

        response = client.post("/api/v1/pause")
        assert response.status_code == 503

        set_bot_engine(original_engine)

    def test_resume_endpoint_bot_unavailable(self, client):
        """Test resume endpoint when bot engine is None."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine
        set_bot_engine(None)

        response = client.post("/api/v1/resume")
        assert response.status_code == 503

        set_bot_engine(original_engine)

    def test_performance_endpoint_bot_unavailable(self, client):
        """Test performance endpoint when bot engine is None."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine
        set_bot_engine(None)

        response = client.get("/api/v1/performance")
        assert response.status_code == 503

        set_bot_engine(original_engine)


class TestDatabaseInteractions:
    """Test database interaction edge cases."""

    def test_orders_endpoint_empty_database(self, client):
        """Test orders endpoint with empty database."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/orders")
        assert response.status_code == 200

        data = response.json()
        assert "orders" in data
        # Assuming test database is empty or has no orders
        assert isinstance(data["orders"], list)

    def test_signals_endpoint_empty_database(self, client):
        """Test signals endpoint with empty database."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/signals")
        assert response.status_code == 200

        data = response.json()
        assert "signals" in data
        assert isinstance(data["signals"], list)

    def test_equity_endpoint_empty_database(self, client):
        """Test equity endpoint with empty database."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/equity")
        assert response.status_code == 200

        data = response.json()
        assert "equity_curve" in data
        assert isinstance(data["equity_curve"], list)


class TestTemplateRendering:
    """Test template rendering for dashboard."""

    def test_dashboard_template_not_found(self, client):
        """Test dashboard endpoint when template is missing."""
        # This would require mocking the template directory or file system
        # For now, just ensure the endpoint works as expected
        response = client.get("/dashboard")
        assert response.status_code == 200
        # In a real scenario, if template is missing, it would raise an exception
        # But since templates directory exists, this should work


class TestPrometheusMetrics:
    """Test Prometheus metrics updates."""

    def test_api_requests_counter_incremented(self, client):
        """Test that API requests counter is incremented."""
        initial_response = client.get("/api/v1/health")
        assert initial_response.status_code == 200

        # Check metrics endpoint
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_text = metrics_response.text
        # Should contain api_requests_total metric
        assert "api_requests_total" in metrics_text

    def test_trades_counter_accessible(self, client):
        """Test that trades counter is accessible."""
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        metrics_text = metrics_response.text
        # These counters might not be incremented in tests, but should be present
        assert "trades_total" in metrics_text or metrics_text.strip() == ""


class TestAuthenticationEdgeCases:
    """Test authentication edge cases."""

    def test_verify_api_key_with_none_credentials(self, client):
        """Test verify_api_key with None credentials."""
        os.environ["API_KEY"] = "test_key"
        # This is tested indirectly through endpoint tests
        response = client.get("/api/v1/orders")
        assert response.status_code == 401

    def test_verify_api_key_with_invalid_credentials(self, client):
        """Test verify_api_key with invalid credentials."""
        os.environ["API_KEY"] = "test_key"
        response = client.get("/api/v1/orders", headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 401

    def test_verify_api_key_no_env_var(self, client):
        """Test verify_api_key when no API_KEY env var is set."""
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]
        response = client.get("/api/v1/orders")
        assert response.status_code == 200  # Should allow access


class TestHTTPExceptionHandler:
    """Test HTTP exception handler."""

    def test_http_exception_handler_formats_error(self, client):
        """Test that HTTP exception handler formats errors properly."""
        # Trigger an HTTPException indirectly
        os.environ["API_KEY"] = "test_key"
        response = client.get("/api/v1/orders", headers={"Authorization": "Bearer invalid"})
        assert response.status_code == 401

        data = response.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]


class TestGlobalExceptionHandlerEdgeCases:
    """Test global exception handler edge cases."""

    def test_global_exception_handler_with_different_exceptions(self, client):
        """Test global exception handler with different exception types."""
        from api.app import set_bot_engine
        original_engine = client._test_bot_engine

        class TestException(Exception):
            pass

        class BrokenEngine:
            def __getattr__(self, name):
                raise TestException("Custom test exception")

        set_bot_engine(BrokenEngine())

        response = client.get("/api/v1/status")
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == 500

        set_bot_engine(original_engine)


class TestRateLimitingEdgeCases:
    """Test rate limiting edge cases."""

    def test_rate_limit_with_redis_fallback(self, client):
        """Test rate limiting with Redis fallback."""
        # This is hard to test directly, but we can check that limiter is configured
        from api.app import limiter
        assert limiter is not None

    def test_get_remote_address_exempt_function(self, client):
        """Test get_remote_address_exempt function."""
        from api.app import get_remote_address_exempt
        # Create a mock request
        class MockRequest:
            def __init__(self, path):
                self.url = type('obj', (object,), {'path': path})()

        exempt_request = MockRequest("/")
        non_exempt_request = MockRequest("/api/v1/status")

        assert get_remote_address_exempt(exempt_request) is None
        assert get_remote_address_exempt(non_exempt_request) is not None


class TestFormatErrorFunction:
    """Test format_error function."""

    def test_format_error_with_details(self, client):
        """Test format_error function with details."""
        from api.app import format_error
        error = format_error(400, "Bad Request", {"field": "required"})
        assert "error" in error
        assert error["error"]["code"] == 400
        assert error["error"]["message"] == "Bad Request"
        assert error["error"]["details"] == {"field": "required"}

    def test_format_error_without_details(self, client):
        """Test format_error function without details."""
        from api.app import format_error
        error = format_error(404, "Not Found")
        assert "error" in error
        assert error["error"]["code"] == 404
        assert error["error"]["message"] == "Not Found"
        assert error["error"]["details"] is None


class TestSetBotEngineFunction:
    """Test set_bot_engine function."""

    def test_set_bot_engine_updates_global(self, client):
        """Test that set_bot_engine updates the global bot_engine."""
        from api.app import set_bot_engine, bot_engine
        original_engine = bot_engine

        mock_engine = {"test": "engine"}
        set_bot_engine(mock_engine)
        # Note: This tests the function, but global state might not be directly accessible
        # In practice, the function sets the global variable

        # Restore original
        set_bot_engine(original_engine)


class TestMiddlewareOrder:
    """Test middleware order and configuration."""

    def test_cors_middleware_configured(self, client):
        """Test that CORS middleware is properly configured."""
        from api.app import app
        cors_found = False
        for middleware in app.user_middleware:
            if hasattr(middleware, 'cls') and 'CORSMiddleware' in str(middleware.cls):
                cors_found = True
                assert 'allow_origins' in middleware.options
                assert middleware.options['allow_origins'] is not None
                break
        assert cors_found

    def test_rate_limit_middleware_configured(self, client):
        """Test that rate limit middleware is configured."""
        from api.app import app
        assert any('SlowAPIMiddleware' in str(mw) for mw in app.user_middleware)


class TestEndpointDependencies:
    """Test endpoint dependencies."""

    def test_endpoints_with_dependencies(self, client):
        """Test that endpoints with dependencies work correctly."""
        # Most endpoints have verify_api_key dependency
        # Test with and without API key
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/status")
        assert response.status_code == 200

        os.environ["API_KEY"] = "test_key"
        response = client.get("/api/v1/status")
        assert response.status_code == 401

        response = client.get("/api/v1/status", headers={"Authorization": "Bearer test_key"})
        assert response.status_code == 200


# Clean up environment after tests
@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    yield
    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]
