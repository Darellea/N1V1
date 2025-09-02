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

    def test_orders_endpoint_without_auth_returns_200(self, client):
        """Test orders endpoint without auth (when API_KEY not set)."""
        # Remove API_KEY for this test
        if "API_KEY" in os.environ:
            del os.environ["API_KEY"]

        response = client.get("/api/v1/orders")
        assert response.status_code == 200

    def test_orders_endpoint_with_auth_returns_401_without_key(self, client):
        """Test orders endpoint with auth enabled but no key provided."""
        os.environ["API_KEY"] = "test_key"
        response = client.get("/api/v1/orders")
        assert response.status_code == 401

    def test_orders_endpoint_with_auth_returns_200_with_key(self, client, auth_headers):
        """Test orders endpoint with auth and correct key."""
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


# Clean up environment after tests
@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    yield
    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]
