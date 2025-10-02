"""
core/idempotency.py

Idempotency management for order execution to prevent duplicate orders.
"""

import uuid
from typing import Any, Callable, Dict, Optional


class OrderExecutionRegistry:
    """
    Registry for tracking order execution states to ensure idempotency.

    This registry stores the execution status of orders by their idempotency key,
    preventing duplicate executions during retries or partial failures.
    """

    def __init__(self):
        """Initialize the registry with an empty dictionary."""
        self._registry: Dict[str, Dict[str, Any]] = {}

    def begin_execution(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Begin execution for the given idempotency key.

        Args:
            key: The idempotency key for the order

        Returns:
            None if execution can proceed (key is new or pending),
            or the state dict if already completed
        """
        if key in self._registry:
            existing = self._registry[key]
            if existing["status"] == "success":
                return existing  # Return the full state dict
            elif existing["status"] == "failure":
                # Allow retry for failed orders
                self._registry[key] = {"status": "pending"}
                return None
            elif existing["status"] == "pending":
                # Block concurrent execution
                return {
                    "status": "pending",
                    "error": "Order execution already in progress",
                }
        else:
            self._registry[key] = {"status": "pending"}
        return None

    def mark_success(self, key: str, result: Dict[str, Any]) -> None:
        """
        Mark the execution as successful.

        Args:
            key: The idempotency key
            result: The execution result to store
        """
        self._registry[key] = {"status": "success", "result": result}

    def mark_failure(self, key: str, error: Exception) -> None:
        """
        Mark the execution as failed.

        Args:
            key: The idempotency key
            error: The exception that occurred
        """
        self._registry[key] = {"status": "failure", "error": str(error)}

    def get_status(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status for an idempotency key.

        Args:
            key: The idempotency key

        Returns:
            The status dictionary or None if not found
        """
        return self._registry.get(key)

    def clear(self) -> None:
        """Clear all entries from the registry (useful for testing)."""
        self._registry.clear()


# Global registry instance (in production, this could be pluggable)
order_execution_registry = OrderExecutionRegistry()


def generate_idempotency_key() -> str:
    """
    Generate a new idempotency key using UUID4.

    Returns:
        A unique idempotency key
    """
    return str(uuid.uuid4())


class RetryNotAllowedError(Exception):
    """
    Exception raised when retry is not allowed for side-effecting operations
    without proper idempotency protection.
    """

    pass


def critical_side_effect(func: Callable) -> Callable:
    """
    Decorator to mark functions as having critical side effects.
    These functions should not be retried without idempotency protection.

    Args:
        func: The function to decorate

    Returns:
        The decorated function with side effect metadata
    """
    func._is_side_effect = True
    return func
