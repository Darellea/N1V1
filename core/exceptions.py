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


class RecoveryError(Exception):
    """
    Raised when a recovery operation fails.

    This exception is raised when systematic error recovery procedures
    fail to restore normal system operation.
    """

    def __init__(
        self,
        message: str,
        component_id: Optional[str] = None,
        failure_type: Optional[str] = None,
        recovery_attempts: int = 0,
    ):
        """
        Initialize RecoveryError.

        Args:
            message: Human-readable error message
            component_id: ID of the component that failed recovery
            failure_type: Type of failure that triggered recovery
            recovery_attempts: Number of recovery attempts made
        """
        super().__init__(message)
        self.component_id = component_id
        self.failure_type = failure_type
        self.recovery_attempts = recovery_attempts

    def __str__(self) -> str:
        """String representation with additional context."""
        msg = super().__str__()
        if self.component_id:
            msg = f"[{self.component_id}] {msg}"
        if self.failure_type:
            msg += f" (failure: {self.failure_type})"
        if self.recovery_attempts > 0:
            msg += f" (attempts: {self.recovery_attempts})"
        return msg


class StateCorruptionError(Exception):
    """
    Raised when system state becomes corrupted and cannot be recovered.

    This exception is raised when critical system state data is found to be
    corrupted and cannot be restored from backups or other recovery mechanisms.
    """

    def __init__(
        self,
        message: str,
        corrupted_fields: Optional[list] = None,
        backup_available: bool = False,
    ):
        """
        Initialize StateCorruptionError.

        Args:
            message: Human-readable error message
            corrupted_fields: List of fields that are corrupted
            backup_available: Whether a backup is available for recovery
        """
        super().__init__(message)
        self.corrupted_fields = corrupted_fields or []
        self.backup_available = backup_available

    def __str__(self) -> str:
        """String representation with additional context."""
        msg = super().__str__()
        if self.corrupted_fields:
            msg += f" (corrupted fields: {', '.join(self.corrupted_fields)})"
        msg += f" (backup available: {self.backup_available})"
        return msg
