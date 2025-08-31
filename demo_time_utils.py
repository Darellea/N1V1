#!/usr/bin/env python3
"""
demo_time_utils.py

Demonstration script showing how to use the timestamp utilities.
"""

import logging
from utils.time import now_ms, to_ms, to_iso


def demo_time_utilities():
    logger = logging.getLogger(__name__)
    logger.info("=== Time Utilities Demo ===")

    # 1. Get current time in milliseconds
    current_ms = now_ms()
    logger.info(f"1. Current time (ms): {current_ms}")

    # 2. Convert to ISO format
    iso_time = to_iso(current_ms)
    logger.info(f"2. ISO format: {iso_time}")

    # 3. Convert back to milliseconds
    round_trip_ms = to_ms(iso_time)
    logger.info(f"3. Round-trip ms: {round_trip_ms}")
    logger.info(f"   Round-trip difference: {abs(round_trip_ms - current_ms)} ms")

    logger.info("=== Conversion Examples ===")

    # Various timestamp formats
    examples = [
        ("Seconds as int", 1672574400),
        ("Milliseconds as int", 1672574400000),
        ("Seconds as float", 1672574400.123),
        ("ISO string", "2023-01-01T12:00:00Z"),
        ("Numeric string seconds", "1672574400"),
        ("Numeric string milliseconds", "1672574400000"),
    ]

    for desc, example in examples:
        result = to_ms(example)
        logger.info(f"{desc:25}: {example} -> {result} ms")

    logger.info("=== Edge Cases ===")

    # Edge cases
    edge_cases = [
        ("None", None),
        ("Invalid string", "invalid"),
        ("List (unsupported)", [1, 2, 3]),
    ]

    for desc, example in edge_cases:
        result = to_ms(example)
        logger.info(f"{desc:25}: {example} -> {result}")

    logger.info("=== ISO Conversion ===")

    # ISO conversion examples
    iso_examples = [
        ("Epoch", 0),
        ("Recent timestamp", current_ms),
        ("Future timestamp", current_ms + 86400000),  # +1 day
    ]

    for desc, example in iso_examples:
        result = to_iso(example)
        logger.info(f"{desc:25}: {example} ms -> {result}")


if __name__ == "__main__":
    import logging
    from utils.logger import setup_logging
    setup_logging()
    demo_time_utilities()
