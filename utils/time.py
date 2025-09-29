"""
utils/time.py

Timestamp utilities for standardizing time handling across the codebase.
Provides functions for working with epoch milliseconds and converting between formats.
"""

from datetime import datetime
from typing import Any, Optional


def now_ms() -> int:
    """
    Get current UTC time in epoch milliseconds.

    Returns:
        int: Current time as epoch milliseconds (UTC)
    """
    return int(datetime.utcnow().timestamp() * 1000)


def to_ms(ts: Any) -> Optional[int]:
    """
    Convert common timestamp inputs to epoch milliseconds.

    Supports conversion from:
    - None -> None
    - int: treated as ms if > 1e12, else seconds (multiplied by 1000)
    - float: same logic as int
    - datetime: converted via .timestamp() * 1000
    - str: try to parse ISO format, fallback to int/float conversion

    Args:
        ts: Timestamp in various formats

    Returns:
        Optional[int]: Epoch milliseconds, or None on failure
    """
    if ts is None:
        return None

    # Handle int and float types
    if isinstance(ts, (int, float)):
        # If value is large (likely already ms), return as is
        if ts > 1e12:  # Threshold for ms timestamps (approx year 2001)
            return int(ts)
        # Otherwise assume seconds and convert to ms
        return int(ts * 1000)

    # Handle datetime objects
    if isinstance(ts, datetime):
        try:
            return int(ts.timestamp() * 1000)
        except (ValueError, TypeError, OSError):
            return None

    # Handle string types
    if isinstance(ts, str):
        # Try ISO format parsing first
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            pass

        # Try parsing as numeric string
        try:
            num = float(ts)
            return to_ms(num)  # Recursively handle numeric conversion
        except (ValueError, TypeError):
            pass

    # Unsupported type or conversion failed
    return None


def to_iso(ts_ms: int) -> str:
    """
    Convert epoch milliseconds to ISO 8601 string.

    Args:
        ts_ms: Epoch milliseconds

    Returns:
        str: ISO 8601 formatted string (UTC)
    """
    try:
        # Convert ms to seconds for datetime
        timestamp_seconds = ts_ms / 1000.0
        dt = datetime.utcfromtimestamp(timestamp_seconds)
        return dt.isoformat() + "Z"
    except (ValueError, TypeError, OSError):
        # Fallback to current time if conversion fails
        return datetime.utcnow().isoformat() + "Z"
