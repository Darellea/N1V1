#!/usr/bin/env python3
"""
Demo script to test the new production-grade logging system.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import (
    generate_correlation_id,
    generate_request_id,
    get_logger_with_context,
    setup_logging,
)


def test_json_logging():
    """Test JSON logging format."""
    print("=== Testing JSON Logging ===")

    # Set environment variables
    os.environ["LOG_FORMAT"] = "json"
    os.environ["LOG_LEVEL"] = "INFO"

    # Setup logging
    logger = setup_logging()

    # Create logger with context
    ctx_logger = get_logger_with_context(
        symbol="BTC/USDT",
        component="TestComponent",
        correlation_id=generate_correlation_id(),
        request_id=generate_request_id(),
        strategy_id="demo_strategy",
    )

    # Log some messages
    ctx_logger.info("Starting test execution")
    ctx_logger.warning("This is a warning message")
    ctx_logger.error("This is an error message")

    print("JSON logging test completed")


def test_pretty_logging():
    """Test pretty logging format."""
    print("\n=== Testing Pretty Logging ===")

    # Set environment variables
    os.environ["LOG_FORMAT"] = "pretty"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Setup logging
    logger = setup_logging()

    # Create logger with context
    ctx_logger = get_logger_with_context(
        symbol="ETH/USDT",
        component="PrettyTest",
        correlation_id=generate_correlation_id(),
        request_id=generate_request_id(),
    )

    # Log some messages
    ctx_logger.debug("Debug message with context")
    ctx_logger.info("Info message with context")
    ctx_logger.warning("Warning message with context")

    print("Pretty logging test completed")


def test_file_logging():
    """Test file logging."""
    print("\n=== Testing File Logging ===")

    # Set environment variables
    os.environ["LOG_FORMAT"] = "json"
    os.environ["LOG_FILE"] = "test_output.log"
    os.environ["LOG_LEVEL"] = "INFO"

    # Setup logging
    logger = setup_logging()

    # Create logger with context
    ctx_logger = get_logger_with_context(
        symbol="ADA/USDT",
        component="FileTest",
        correlation_id=generate_correlation_id(),
        request_id=generate_request_id(),
        strategy_id="file_test_strategy",
    )

    # Log some messages
    ctx_logger.info("Message 1 for file logging")
    ctx_logger.info("Message 2 for file logging")
    ctx_logger.error("Error message for file logging")

    # Check if file was created and contains JSON
    if Path("test_output.log").exists():
        print("Log file created successfully")
        with open("test_output.log", "r") as f:
            content = f.read()
            print(f"Log file content length: {len(content)} characters")

            # Try to parse first line as JSON
            lines = content.strip().split("\n")
            if lines:
                try:
                    parsed = json.loads(lines[0])
                    print(f"First log entry parsed successfully: {parsed['message']}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON: {e}")
    else:
        print("Log file was not created")

    print("File logging test completed")


if __name__ == "__main__":
    print("Testing Production-Grade Logging System")
    print("=" * 50)

    test_json_logging()
    test_pretty_logging()
    test_file_logging()

    print("\n" + "=" * 50)
    print("All tests completed!")

    # Clean up
    if Path("test_output.log").exists():
        Path("test_output.log").unlink()
        print("Cleaned up test log file")
