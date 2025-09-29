#!/usr/bin/env python3
"""
demo_time_utils.py

Comprehensive demonstration of timestamp utilities with robust error handling.
Provides functions for converting between various timestamp formats and logging utilities.
"""

import contextlib
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

# Module-level logger
logger = logging.getLogger(__name__)


def now_ms() -> int:
    """
    Get current UTC time in epoch milliseconds.

    Returns:
        int: Current time as epoch milliseconds (UTC)
    """
    return int(time.time() * 1000)


def to_ms(ts: Any) -> Optional[int]:
    """
    Convert various timestamp formats to epoch milliseconds.

    Supports:
    - int/float: treated as seconds if < 1e12, milliseconds if >= 1e12
    - str: ISO format or numeric strings
    - datetime: converted via timestamp()
    - None: returns None

    Args:
        ts: Input timestamp in various formats

    Returns:
        Optional[int]: Epoch milliseconds, or None if conversion fails

    Raises:
        ValueError: If input type is not supported (list, dict, etc.)
    """
    if ts is None:
        return None

    # Reject unsupported types that could cause TypeError
    if isinstance(ts, (list, dict, set, tuple)):
        raise ValueError(
            f"Unsupported input type: {type(ts).__name__}. Expected int, float, str, datetime, or None."
        )

    # Handle numeric types
    if isinstance(ts, (int, float)):
        # If value is large (likely already ms), return as is
        if ts > 1e12:  # Threshold for ms timestamps (approx year 2001)
            return int(ts)
        # Otherwise assume seconds and convert to ms
        return int(ts * 1000)

    # Handle datetime objects
    if isinstance(ts, datetime):
        try:
            # Ensure UTC timezone
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return int(ts.timestamp() * 1000)
        except (ValueError, TypeError, OSError, AttributeError):
            return None

    # Handle string types
    if isinstance(ts, str):
        # Try ISO format parsing first
        try:
            # Handle various ISO formats
            ts_clean = ts.replace("Z", "+00:00")
            if not ts_clean.endswith("+00:00"):
                ts_clean += "+00:00"
            dt = datetime.fromisoformat(ts_clean)
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError, AttributeError):
            pass

        # Try parsing as numeric string
        try:
            num = float(ts)
            return to_ms(num)  # Recursively handle numeric conversion
        except (ValueError, TypeError, AttributeError):
            pass

    # Unsupported type or conversion failed
    return None


def to_iso(ts_ms: Any) -> str:
    """
    Convert epoch milliseconds to ISO 8601 string (UTC).

    Args:
        ts_ms: Epoch milliseconds (int, float, or str that can be converted)

    Returns:
        str: ISO 8601 formatted string (UTC)

    Raises:
        ValueError: If input type is not supported
    """
    # First convert to milliseconds if needed
    if not isinstance(ts_ms, (int, float)):
        ts_ms = to_ms(ts_ms)
        if ts_ms is None:
            # Fallback to current time if conversion fails
            ts_ms = now_ms()

    try:
        # Convert ms to seconds for datetime
        timestamp_seconds = ts_ms / 1000.0
        dt = datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except (ValueError, TypeError, OSError, AttributeError):
        # Fallback to current time if conversion fails
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def to_datetime(ts: Any) -> Optional[datetime]:
    """
    Convert various timestamp formats to datetime object (UTC).

    Args:
        ts: Input timestamp in various formats

    Returns:
        Optional[datetime]: Datetime object in UTC timezone, or None if conversion fails

    Raises:
        ValueError: If input type is not supported
    """
    ms = to_ms(ts)
    if ms is None:
        return None

    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    except (ValueError, TypeError, OSError, AttributeError):
        return None


def log_current_time(message: str = "Current time") -> None:
    """
    Log current time with a custom message.

    Args:
        message: Custom message prefix
    """
    current_ms = now_ms()
    iso_time = to_iso(current_ms)
    logger.info(f"{message}: {iso_time} ({current_ms} ms)")


def log_conversion(input_val: Any, description: str = "Conversion") -> None:
    """
    Log a timestamp conversion with description.

    Args:
        input_val: Input value to convert
        description: Description of the conversion
    """
    try:
        result_ms = to_ms(input_val)
        result_iso = to_iso(result_ms) if result_ms is not None else "N/A"
        logger.info(f"{description}: {input_val} -> {result_iso} ({result_ms} ms)")
    except ValueError as e:
        logger.warning(f"{description}: {input_val} -> ERROR: {e}")


@contextlib.contextmanager
def safe_file_logging(logfile_path: str, level: int = logging.INFO):
    """
    Context manager for safe file logging with automatic cleanup.

    Args:
        logfile_path: Path to log file
        level: Logging level
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(logfile_path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)
    logger.setLevel(level)

    try:
        yield logger
    finally:
        # Clean up
        logger.removeHandler(file_handler)
        file_handler.close()


def demo_time_utilities_basic_execution():
    """Basic execution demo showing core functionality."""
    # No header for basic execution

    # Get current time
    current_ms = now_ms()
    logger.info(f"Current time (ms): {current_ms}")

    # Convert to ISO
    iso_time = to_iso(current_ms)
    logger.info(f"ISO format: {iso_time}")

    # Round-trip conversion
    round_trip_ms = to_ms(iso_time)
    logger.info(f"Round-trip ms: {round_trip_ms}")
    if round_trip_ms is not None:
        logger.info(f"Round-trip difference: {abs(round_trip_ms - current_ms)} ms")


def demo_time_utilities_current_time_logging():
    """Demo of current time logging functionality."""
    # No header

    log_current_time("Application started at")
    time.sleep(0.001)  # Small delay for demonstration
    log_current_time("Application running at")


def demo_time_utilities_conversion_examples():
    """Demo of various timestamp conversion examples."""
    logger.info("=== Conversion Examples ===")

    examples = [
        (1672574400, "Seconds as int"),
        (1672574400000, "Milliseconds as int"),
        (1672574400.123, "Seconds as float"),
        ("2023-01-01T12:00:00Z", "ISO string"),
        ("1672574400", "Numeric string seconds"),
        ("1672574400000", "Numeric string milliseconds"),
    ]

    for value, description in examples:
        log_conversion(value, description)


def demo_time_utilities_edge_cases():
    """Demo of edge cases and error handling."""
    logger.info("=== Edge Cases ===")

    edge_cases = [
        (None, "None value"),
        ("invalid", "Invalid string"),
        ("", "Empty string"),
        (0, "Zero timestamp"),
        (-1, "Negative timestamp"),
    ]

    for value, description in edge_cases:
        try:
            log_conversion(value, description)
        except ValueError as e:
            logger.warning(f"{description}: {value} -> ValueError: {e}")

    # Test unsupported types (should raise ValueError)
    unsupported_cases = [
        ([1, 2, 3], "List (unsupported)"),
        ({"key": "value"}, "Dict (unsupported)"),
        ({1, 2, 3}, "Set (unsupported)"),
        ((1, 2, 3), "Tuple (unsupported)"),
    ]

    for value, description in unsupported_cases:
        try:
            log_conversion(value, description)
        except ValueError as e:
            logger.warning(f"{description}: {value} -> ValueError: {e}")


def demo_time_utilities_iso_examples():
    """Demo of ISO conversion examples."""
    logger.info("=== ISO Conversion ===")

    iso_examples = [
        (0, "Epoch"),
        (now_ms(), "Recent timestamp"),
        (now_ms() + 86400000, "Future timestamp"),
    ]

    for value, description in iso_examples:
        log_conversion(value, description)


def demo_time_utilities():
    """Main demo function that runs all demonstrations."""
    logger.info("=== Time Utilities Demo ===")

    try:
        demo_time_utilities_basic_execution()
        demo_time_utilities_current_time_logging()
        demo_time_utilities_conversion_examples()
        demo_time_utilities_edge_cases()
        demo_time_utilities_iso_examples()

        logger.info("=== Demo completed successfully ===")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        # Do not raise to avoid crashing


def main():
    """Main entry point when run as script."""
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run demo with safe file logging
    with safe_file_logging("./demo_time_utils.log"):
        demo_time_utilities()


if __name__ == "__main__":
    main()
