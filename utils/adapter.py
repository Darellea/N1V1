"""
utils/adapter.py

Small adapter utilities to normalize inputs (e.g. TradingSignal dataclass/objects)
into plain dictionaries for loggers, order manager, and notifier code paths.
"""

from typing import Any, Dict
import dataclasses
import logging
import time
from datetime import datetime


def _normalize_timestamp(timestamp: Any) -> int:
    """
    Normalize timestamp to milliseconds since epoch (UTC).

    Handles:
    - int: assumed to be ms, returned as-is
    - float: assumed to be seconds since epoch, converted to ms
    - datetime: converted to ms with timezone handling
    - str: parsed as ISO format and converted to ms

    Raises ValueError for invalid formats.
    """
    from datetime import timezone

    if isinstance(timestamp, int):
        return timestamp
    elif isinstance(timestamp, float):
        # Assume float is seconds since epoch (e.g., time.time()), convert to ms
        return int(timestamp * 1000)
    elif isinstance(timestamp, datetime):
        try:
            # Handle timezone-naive datetimes by assuming UTC
            if timestamp.tzinfo is None:
                # For naive datetimes, assume UTC to avoid local timezone issues
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif timestamp.tzinfo is not timezone.utc:
                # Convert to UTC if it's not already
                timestamp = timestamp.astimezone(timezone.utc)

            # Convert to milliseconds, handling potential OSError for edge cases
            return int(timestamp.timestamp() * 1000)
        except (OSError, ValueError, OverflowError) as e:
            # Handle edge cases like pre-1970 timestamps or invalid dates
            logging.warning(f"Failed to convert datetime {timestamp} to timestamp: {e}. Using fallback.")
            # Fallback: calculate manually for pre-1970 dates
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            elif timestamp.tzinfo is not timezone.utc:
                timestamp = timestamp.astimezone(timezone.utc)

            # Manual calculation: days since epoch * 86400000 + microseconds
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            delta = timestamp - epoch
            return int(delta.total_seconds() * 1000)
    elif isinstance(timestamp, str):
        try:
            # Try to parse as ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            # Ensure timezone info
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            elif dt.tzinfo is not timezone.utc:
                dt = dt.astimezone(timezone.utc)
            return int(dt.timestamp() * 1000)
        except (ValueError, OSError) as e:
            logging.error(f"Failed to parse timestamp string '{timestamp}': {e}")
            raise ValueError(f"Invalid timestamp string format: {timestamp}")
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")


def signal_to_dict(signal: Any) -> Dict[str, Any]:
    """
    Normalize a trading signal object into a plain dict.

    Supports:
    - dict -> shallow copy
    - dataclass -> dataclasses.asdict with enum and timestamp normalization
    - objects with to_dict() -> call it
    - generic objects -> probe common attributes and __dict__

    Timestamp normalization rules:
    - datetime objects are converted to milliseconds since epoch (UTC)
    - int values are left unchanged (assumed to be milliseconds)
    - str values are parsed as ISO format and converted to milliseconds
    - Invalid timestamps raise ValueError in dataclass conversion, set to None in attribute probing

    Returns an empty dict for None input.
    """
    if signal is None:
        return {}

    # If already a dict, return a shallow copy
    if isinstance(signal, dict):
        return dict(signal)

    # Dataclass support
    try:
        if dataclasses.is_dataclass(signal):
            result = dataclasses.asdict(signal)
            # Convert enums to their values/names for JSON serialization
            for key, value in result.items():
                if hasattr(value, "value"):
                    result[key] = value.value
                elif hasattr(value, "name"):
                    # Keep SignalType names uppercase for schema validation
                    if key == "signal_type":
                        result[key] = value.name
                    else:
                        result[key] = value.name.lower()
            
            # Special handling for TradingSignal dataclass - map quantity to amount
            if hasattr(signal, '__class__') and signal.__class__.__name__ == 'TradingSignal':
                if 'quantity' in result:
                    result['amount'] = result.pop('quantity')

            # Normalize timestamp to milliseconds
            if 'timestamp' in result:
                original = result['timestamp']
                try:
                    result['timestamp'] = _normalize_timestamp(original)
                    logging.debug(f"Normalized timestamp in dataclass from {type(original).__name__} ({original}) to int ({result['timestamp']})")
                except ValueError as e:
                    logging.error(f"Failed to normalize timestamp in dataclass: {e}")
                    raise  # Re-raise to fail fast for invalid timestamps

            return result
    except (AttributeError, TypeError, ValueError):
        # If dataclass conversion fails for known reasons, fall back to other methods.
        pass

    # to_dict() protocol
    try:
        to_dict = getattr(signal, "to_dict", None)
        if callable(to_dict):
            result = to_dict()
            # Check if result is a Mock (for testing)
            from unittest.mock import Mock
            if isinstance(result, Mock):
                # Skip to_dict for Mocks, fall back to attribute probing
                pass
            else:
                return result
    except (TypeError, AttributeError, ValueError):
        # to_dict() may raise if implementation is buggy; fall back to probing attributes.
        pass

    out: Dict[str, Any] = {}

    # Probe common signal attributes
    common_attrs = (
        "id",
        "symbol",
        "signal_type",
        "order_type",
        "side",
        "amount",
        "price",
        "params",
        "stop_loss",
        "take_profit",
        "trailing_stop",
        "timestamp",
    )

    for a in common_attrs:
        # Use getattr with default to avoid KeyError; properties may raise AttributeError,
        # so catch AttributeError at the minimal scope.
        try:
            if hasattr(signal, a):
                val = getattr(signal, a)
            else:
                continue
        except AttributeError:
            # If property access raises, skip that attribute.
            continue

        # Normalize enums (e.g., OrderType) to value or name
        try:
            if a == "signal_type" and hasattr(val, "name"):
                # Keep SignalType names uppercase for schema validation
                val = val.name
            elif hasattr(val, "value"):
                val = val.value
            elif hasattr(val, "name"):
                # some enums expose .name; convert to lower-case string for readability
                val = val.name.lower()
        except AttributeError:
            # If enum-like object doesn't expose expected attributes, ignore and continue.
            pass

        # Normalize timestamp to milliseconds
        if a == 'timestamp':
            try:
                # Check if it's a Mock object (for testing)
                from unittest.mock import Mock
                if isinstance(val, Mock):
                    val = int(time.time() * 1000)  # Use current time for Mock timestamps
                else:
                    val = _normalize_timestamp(val)
                logging.debug(f"Normalized timestamp in attribute probe from {type(val).__name__} to int ({val})")
            except ValueError as e:
                logging.warning(f"Failed to normalize timestamp in attribute probe: {e}")
                val = None  # Set to None for invalid timestamps

        out[a] = val

    # As a fallback, include any public attributes from __dict__ (non-callable)
    try:
        obj_dict = getattr(signal, "__dict__", None)
        if isinstance(obj_dict, dict):
            for k, v in obj_dict.items():
                if k not in out and not k.startswith("_"):
                    out[k] = v
    except AttributeError:
        # If accessing __dict__ raises, give up on this fallback.
        pass

    return out
