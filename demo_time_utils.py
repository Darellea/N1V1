#!/usr/bin/env python3
"""
demo_time_utils.py

Demonstration script showing how to use the timestamp utilities.
"""

import sys
from datetime import datetime, timezone
from utils.time import now_ms, to_ms, to_iso


def demo_time_utilities():
    """Demonstrate the time utility functions"""
    print("=== Time Utilities Demo ===\n")
    
    # 1. Get current time in milliseconds
    current_ms = now_ms()
    print(f"1. Current time (ms): {current_ms}")
    
    # 2. Convert to ISO format
    iso_time = to_iso(current_ms)
    print(f"2. ISO format: {iso_time}")
    
    # 3. Convert back to milliseconds
    round_trip_ms = to_ms(iso_time)
    print(f"3. Round-trip ms: {round_trip_ms}")
    print(f"   Round-trip difference: {abs(round_trip_ms - current_ms)} ms")
    
    print("\n=== Conversion Examples ===\n")
    
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
        print(f"{desc:25}: {example} -> {result} ms")
    
    print("\n=== Edge Cases ===\n")
    
    # Edge cases
    edge_cases = [
        ("None", None),
        ("Invalid string", "invalid"),
        ("List (unsupported)", [1, 2, 3]),
    ]
    
    for desc, example in edge_cases:
        result = to_ms(example)
        print(f"{desc:25}: {example} -> {result}")
    
    print("\n=== ISO Conversion ===\n")
    
    # ISO conversion examples
    iso_examples = [
        ("Epoch", 0),
        ("Recent timestamp", current_ms),
        ("Future timestamp", current_ms + 86400000),  # +1 day
    ]
    
    for desc, example in iso_examples:
        result = to_iso(example)
        print(f"{desc:25}: {example} ms -> {result}")


if __name__ == "__main__":
    demo_time_utilities()
