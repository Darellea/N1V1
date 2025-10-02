"""
Custom exceptions for the N1V1 trading framework.
"""

from typing import Any, Dict, Optional


class SchemaValidationError(Exception):
    """
    Raised when inbound data fails schema validation.

    This exception is raised when external data (API responses, WebSocket messages,
    market data feeds, etc.) does not conform to expected schemas.
    """

    def __init__(
        self,
        message: str,
        data: Optional[Any] = None,
        schema_name: Optional[str] = None,
        field_errors: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize SchemaValidationError.

        Args:
            message: Human-readable error message
            data: The invalid data that was rejected
            schema_name: Name of the schema that was violated
            field_errors: Detailed field-level validation errors
        """
        super().__init__(message)
        self.data = data
        self.schema_name = schema_name
        self.field_errors = field_errors or {}

    def __str__(self) -> str:
        """String representation with additional context."""
        msg = super().__str__()
        if self.schema_name:
            msg = f"[{self.schema_name}] {msg}"
        if self.field_errors:
            msg += f" (field errors: {self.field_errors})"
        return msg


class MissingIdempotencyError(Exception):
    """
    Raised when an idempotency key is required but not provided.

    This exception is raised when attempting to execute an order without
    providing an idempotency key, which is required for safe retries.
    """

    def __init__(
        self, message: str = "Idempotency key is required for order execution"
    ):
        super().__init__(message)
