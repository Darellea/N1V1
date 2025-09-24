"""
Custom middleware for the FastAPI application.
"""

from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import logging


class RateLimitException(Exception):
    """Custom exception for rate limiting."""
    def __init__(self, headers=None):
        super().__init__("Rate limit exceeded")
        self.headers = headers or {}


class RateLimitExceptionMiddleware(BaseHTTPMiddleware):
    """Middleware to convert SlowAPI 429 responses to JSON responses."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)

        if response.status_code == 429:
            # Extract headers from response
            headers = {}
            for key, value in response.headers.items():
                headers[key] = value

            # Return JSON response directly
            from .app import rate_limit_exception_handler
            import asyncio
            # Since we can't call async handler here, create the response directly
            json_response = {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Rate limit exceeded",
                    "details": {
                        "limit": 60,
                        "window": "1 minute",
                        "endpoint": str(request.url.path)
                    }
                }
            }
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=429, content=json_response, headers=headers)

        return response


class CustomExceptionMiddleware(ExceptionMiddleware):
    """Custom exception middleware to properly handle exceptions."""

    async def __call__(self, scope, receive, send):
        try:
            await self.app(scope, receive, send)
        except RateLimitException:
            # Let RateLimitException be handled by the built-in ExceptionMiddleware
            raise
        except Exception as exc:
            # Check if this is an HTTP request
            if scope["type"] == "http":
                # Create a minimal request object
                request = Request(scope, receive)

                # Get the global exception handler
                from .app import global_exception_handler
                response = await global_exception_handler(request, exc)

                # Send the response
                await response(scope, receive, send)
            else:
                raise
