"""
Execution Types Module

Contains common enums and types used across the execution layer to avoid circular imports.
"""

from enum import Enum


class ExecutionPolicy(Enum):
    """Available execution policies."""

    TWAP = "twap"
    VWAP = "vwap"
    DCA = "dca"
    SMART_SPLIT = "smart_split"
    MARKET = "market"
    LIMIT = "limit"


class ExecutionStatus(Enum):
    """Execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    FALLBACK = "fallback"
