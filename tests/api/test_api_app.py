"""
Tests for api/app.py - FastAPI application and middleware.
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.responses import JSONResponse


class TestCustomExceptionMiddleware:
    """Test the CustomExceptionMiddleware class."""

    @pytest.fixture
    def middleware(self):
        """Create a CustomExceptionMiddleware instance."""
        from api.app import CustomExceptionMiddleware

        # Create a mock app
        mock_app = Mock()
        middleware = CustomExceptionMiddleware(mock_app)
        return middleware

    @pytest.fixture
    def mock_scope_http(self):
        """Create a mock HTTP scope."""
        return {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
        }

    @pytest.fixture
    def mock_scope_websocket(self):
        """Create a mock WebSocket scope."""
        return {
            "type": "websocket",
            "path": "/ws",
            "query_string": b"",
            "headers": [],
        }

    @pytest.mark.asyncio
    async def test_non_http_exception_reraised(self, middleware, mock_scope_websocket):
        """Test that non-HTTP exceptions are re-raised."""
        # Mock the receive and send functions
        receive = AsyncMock()
        send = AsyncMock()

        # Mock the app to raise an exception
        test_exception = ValueError("Test exception")
        middleware.app = AsyncMock(side_effect=test_exception)

        # Call the middleware - should re-raise the exception
        with pytest.raises(ValueError, match="Test exception"):
            await middleware(mock_scope_websocket, receive, send)

    @pytest.mark.asyncio
    async def test_normal_request_passthrough(self, middleware, mock_scope_http):
        """Test that normal requests pass through without issues."""
        # Mock the receive and send functions
        receive = AsyncMock()
        send = AsyncMock()

        # Mock the app to work normally
        middleware.app = AsyncMock()

        # Call the middleware
        await middleware(mock_scope_http, receive, send)

        # Verify the app was called
        middleware.app.assert_called_once_with(mock_scope_http, receive, send)

        # Verify send was called
        send.assert_called_once()


class TestRateLimitJSONMiddleware:
    """Test the RateLimitJSONMiddleware class."""

    @pytest.fixture
    def middleware(self):
        """Create a RateLimitJSONMiddleware instance."""
        from api.app import RateLimitJSONMiddleware

        # Create a mock app
        mock_app = Mock()
        middleware = RateLimitJSONMiddleware(mock_app)
        return middleware

    @pytest.mark.asyncio
    async def test_rate_limit_response_conversion(self, middleware):
        """Test that 429 responses are converted to JSON."""
        # Mock request
        request = Mock()
        request.url.path = "/api/v1/status"

        # Mock response with 429 status
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"X-RateLimit-Limit": "60"}
        mock_response.body_iterator = AsyncMock()
        mock_response.body_iterator.__aiter__.return_value = [b"Rate limit exceeded"]

        # Mock the call_next function
        call_next = AsyncMock(return_value=mock_response)

        # Call the middleware
        response = await middleware.dispatch(request, call_next)

        # Verify it's a JSONResponse
        assert isinstance(response, JSONResponse)
        assert response.status_code == 429

        # Verify the content structure
        content = response.body
        response_data = json.loads(content.decode("utf-8"))

        assert "error" in response_data
        assert response_data["error"]["code"] == "rate_limit_exceeded"
        assert "details" in response_data["error"]
        assert response_data["error"]["details"]["endpoint"] == "/api/v1/status"

    @pytest.mark.asyncio
    async def test_normal_response_passthrough(self, middleware):
        """Test that normal responses pass through unchanged."""
        # Mock request
        request = Mock()

        # Mock normal response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {}

        # Mock the call_next function
        call_next = AsyncMock(return_value=mock_response)

        # Call the middleware
        response = await middleware.dispatch(request, call_next)

        # Verify the response is unchanged
        assert response is mock_response


class TestExceptionHandlers:
    """Test the exception handlers."""

    @pytest.mark.asyncio
    async def test_http_exception_handler_formatted_error(self):
        """Test HTTP exception handler with already formatted error."""
        from fastapi import HTTPException

        from api.app import http_exception_handler

        # Mock request
        request = Mock()

        # Create HTTP exception with formatted error
        formatted_error = {"error": {"code": 401, "message": "Invalid API key"}}
        exc = HTTPException(status_code=401, detail=formatted_error)

        # Call the handler
        response = await http_exception_handler(request, exc)

        # Verify it's a JSONResponse
        assert isinstance(response, JSONResponse)
        assert response.status_code == 401

        # Verify the content is the formatted error
        content = response.body
        response_data = json.loads(content.decode("utf-8"))
        assert response_data == formatted_error

    @pytest.mark.asyncio
    async def test_http_exception_handler_plain_error(self):
        """Test HTTP exception handler with plain string error."""
        from fastapi import HTTPException

        from api.app import http_exception_handler

        # Mock request
        request = Mock()

        # Create HTTP exception with plain string
        exc = HTTPException(status_code=404, detail="Not found")

        # Call the handler
        response = await http_exception_handler(request, exc)

        # Verify it's a JSONResponse
        assert isinstance(response, JSONResponse)
        assert response.status_code == 404

        # Verify the content structure
        content = response.body
        response_data = json.loads(content.decode("utf-8"))

        assert "error" in response_data
        assert response_data["error"]["code"] == 404
        assert response_data["error"]["message"] == "Not found"

    @pytest.mark.asyncio
    async def test_global_exception_handler(self):
        """Test the global exception handler."""
        from api.app import global_exception_handler

        # Mock request
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/v1/status"

        # Create a test exception
        exc = ValueError("Test exception")

        # Call the handler
        response = await global_exception_handler(request, exc)

        # Verify it's a JSONResponse
        assert isinstance(response, JSONResponse)
        assert response.status_code == 500

        # Verify the content structure
        content = response.body
        response_data = json.loads(content.decode("utf-8"))

        assert "error" in response_data
        assert response_data["error"]["code"] == 500
        assert "An unexpected error occurred" in response_data["error"]["message"]
        assert response_data["error"]["details"]["path"] == "/api/v1/status"
        assert response_data["error"]["details"]["method"] == "GET"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_format_error_without_details(self):
        """Test format_error function without details."""
        from api.app import format_error

        result = format_error(404, "Not found")

        expected = {"error": {"code": 404, "message": "Not found", "details": None}}

        assert result == expected

    def test_format_error_with_details(self):
        """Test format_error function with details."""
        from api.app import format_error

        details = {"endpoint": "/api/v1/status"}
        result = format_error(401, "Invalid API key", details)

        expected = {
            "error": {"code": 401, "message": "Invalid API key", "details": details}
        }

        assert result == expected

    def test_get_remote_address_exempt_root(self):
        """Test get_remote_address_exempt for root endpoint."""
        from api.app import get_remote_address_exempt

        # Mock request for root endpoint
        request = Mock()
        request.url.path = "/"

        result = get_remote_address_exempt(request)

        # Should return None (exempt from rate limiting)
        assert result is None

    def test_get_remote_address_exempt_dashboard(self):
        """Test get_remote_address_exempt for dashboard endpoint."""
        from api.app import get_remote_address_exempt

        # Mock request for dashboard endpoint
        request = Mock()
        request.url.path = "/dashboard"

        result = get_remote_address_exempt(request)

        # Should return None (exempt from rate limiting)
        assert result is None

    def test_set_bot_engine(self):
        """Test set_bot_engine function."""
        from api.app import set_bot_engine

        # Create a mock bot engine
        mock_engine = Mock()

        # Set the bot engine
        set_bot_engine(mock_engine)

        # Import bot_engine again to check if it was set
        from api.app import bot_engine as current_bot_engine

        assert current_bot_engine is mock_engine


class TestAPIEndpoints:
    """Test API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from fastapi.testclient import TestClient

        from api.app import app

        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Crypto Trading Bot API"
        assert data["version"] == "1.0.0"

    def test_health_endpoint_no_bot_engine(self, client):
        """Test health endpoint when bot engine is not available."""
        # Ensure bot_engine is None
        from api.app import set_bot_engine

        set_bot_engine(None)

        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "Bot engine not available" in data["detail"]

    def test_metrics_endpoint(self, client):
        """Test the metrics endpoint."""
        response = client.get("/metrics")

        # Should return some content (Prometheus format)
        assert response.status_code == 200
        assert "text/plain" in str(response.headers.get("content-type", ""))

    @pytest.mark.asyncio
    async def test_verify_api_key_no_env_var(self):
        """Test verify_api_key when no API_KEY env var is set."""
        from api.app import verify_api_key

        # Ensure API_KEY is not set
        with patch.dict("os.environ", {}, clear=True):
            result = await verify_api_key(None)
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_api_key_valid(self):
        """Test verify_api_key with valid credentials."""
        from fastapi.security import HTTPAuthorizationCredentials

        from api.app import verify_api_key

        # Mock credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="test_key"
        )

        # Set API_KEY env var
        with patch.dict("os.environ", {"API_KEY": "test_key"}):
            result = await verify_api_key(credentials)
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_api_key_invalid(self):
        """Test verify_api_key with invalid credentials."""
        from fastapi.security import HTTPAuthorizationCredentials

        from api.app import HTTPException, verify_api_key

        # Mock credentials
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="wrong_key"
        )

        # Set API_KEY env var
        with patch.dict("os.environ", {"API_KEY": "test_key"}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(credentials)

            assert exc_info.value.status_code == 401
            assert "Invalid API key" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_verify_api_key_required_but_missing(self):
        """Test verify_api_key when API key is required but not provided."""
        from api.app import HTTPException, verify_api_key

        # Set API_KEY env var but don't provide credentials
        with patch.dict("os.environ", {"API_KEY": "test_key"}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(None)

            assert exc_info.value.status_code == 401
            assert "API key required" in str(exc_info.value.detail)
