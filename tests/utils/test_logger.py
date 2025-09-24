"""
Comprehensive tests for utils/logger.py

Covers logging setup, trade logging, performance tracking, and error scenarios.
Tests specific lines: 112-131, 139-141, 152-153, 161-167, 175-180, 187-192,
199-204, 235-236, 267-281, 291-296, 300, 307-317, 321-325, 431, 442-452, 463-479.
"""

import pytest
import logging
import json
import csv
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import uuid
from datetime import datetime
from typing import Dict, Any

from utils.logger import (
    TradeLogger,
    setup_logging,
    get_trade_logger,
    generate_correlation_id,
    generate_request_id,
    get_logger_with_context,
    log_to_file,
    ColorFormatter,
    JSONFormatter,
    PrettyFormatter,
    LOGS_DIR,
    TRADE_LEVEL,
    PERF_LEVEL
)


class TestColorFormatter:
    """Test cases for ColorFormatter class (lines 112-131)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ColorFormatter()

    def test_format_debug_level(self):
        """Test formatting DEBUG level messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Debug message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Debug message" in formatted
        # Should contain cyan color code
        assert "\x1b[36m" in formatted

    def test_format_info_level(self):
        """Test formatting INFO level messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Info message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Info message" in formatted
        # Should contain green color code
        assert "\x1b[32m" in formatted

    def test_format_warning_level(self):
        """Test formatting WARNING level messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Warning message" in formatted
        # Should contain yellow color code
        assert "\x1b[33m" in formatted

    def test_format_error_level(self):
        """Test formatting ERROR level messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Error message" in formatted
        # Should contain red color code
        assert "\x1b[31m" in formatted

    def test_format_critical_level(self):
        """Test formatting CRITICAL level messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.CRITICAL,
            pathname="",
            lineno=0,
            msg="Critical message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Critical message" in formatted
        # Should contain white on red color codes
        assert "\x1b[37m\x1b[41m" in formatted

    def test_format_trade_level(self):
        """Test formatting TRADE level messages."""
        record = logging.LogRecord(
            name="test",
            level=TRADE_LEVEL,
            pathname="",
            lineno=0,
            msg="Trade message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Trade message" in formatted
        # Should contain magenta color code
        assert "\x1b[35m" in formatted

    def test_format_perf_level(self):
        """Test formatting PERF level messages."""
        record = logging.LogRecord(
            name="test",
            level=PERF_LEVEL,
            pathname="",
            lineno=0,
            msg="Performance message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Performance message" in formatted
        # Should contain blue color code
        assert "\x1b[34m" in formatted

    def test_format_unknown_level(self):
        """Test formatting unknown level messages."""
        record = logging.LogRecord(
            name="test",
            level=999,
            pathname="",
            lineno=0,
            msg="Unknown level message",
            args=(),
            exc_info=None
        )
        formatted = self.formatter.format(record)
        assert "Unknown level message" in formatted
        # Should not contain color codes for unknown levels
        assert not any(color in formatted for color in ["\x1b[3", "\x1b[4"])


class TestTradeLogger:
    """Test cases for TradeLogger class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = TradeLogger("test_logger")

    def teardown_method(self):
        """Clean up after each test."""
        # Clean up any created log files
        if hasattr(self.logger, 'trade_csv') and self.logger.trade_csv.exists():
            try:
                self.logger.trade_csv.unlink()
            except:
                pass

    def test_init_trade_logger(self):
        """Test TradeLogger initialization (lines 139-141)."""
        assert self.logger.name == "test_logger"
        assert hasattr(self.logger, 'trades')
        assert hasattr(self.logger, 'performance_stats')
        assert hasattr(self.logger, 'trade_csv')
        assert isinstance(self.logger.trades, list)
        assert isinstance(self.logger.performance_stats, dict)

    def test_init_trade_csv_creates_file(self):
        """Test _init_trade_csv creates CSV file with header (lines 152-153)."""
        # Clean up any existing file
        if self.logger.trade_csv.exists():
            self.logger.trade_csv.unlink()

        # Re-initialize
        self.logger._init_trade_csv()

        assert self.logger.trade_csv.exists()

        # Check header
        with open(self.logger.trade_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_header = ["timestamp", "pair", "action", "size", "entry_price", "exit_price", "pnl"]
            assert header == expected_header

    def test_init_trade_csv_handles_io_error(self):
        """Test _init_trade_csv handles I/O errors gracefully."""
        with patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")):
            # Should not raise exception
            self.logger._init_trade_csv()

    def test_trade_method_logs_message(self):
        """Test trade method logs messages (lines 161-167)."""
        with patch.object(self.logger, 'log') as mock_log:
            with patch.object(self.logger, '_record_trade') as mock_record:
                trade_data = {"pair": "BTC/USDT", "action": "BUY"}
                extra = {"symbol": "BTC/USDT"}

                self.logger.trade("Test trade", trade_data, extra=extra)

                # Verify log was called with correct level and message
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[0][0] == TRADE_LEVEL  # 21
                assert call_args[0][1] == "Test trade"
                assert call_args[1]['extra'] == extra

                # Verify _record_trade was called
                mock_record.assert_called_once()

    def test_trade_method_handles_type_error(self):
        """Test trade method handles TypeError gracefully."""
        # Mock the log method to simulate old Python logging behavior
        original_log = self.logger.log
        def mock_log(level, msg, *args, **kwargs):
            if 'extra' in kwargs:
                raise TypeError("Old Python logging")
            return original_log(level, msg, *args, **kwargs)

        with patch.object(self.logger, 'log', mock_log):
            with patch.object(self.logger, '_record_trade') as mock_record:
                trade_data = {"pair": "BTC/USDT"}

                # Should not raise exception
                self.logger.trade("Test trade", trade_data)

                # Should still call _record_trade
                mock_record.assert_called_once()

    def test_performance_method_logs_metrics(self):
        """Test performance method logs metrics (lines 175-180)."""
        with patch.object(self.logger, 'log') as mock_log:
            with patch.object(self.logger, '_update_performance') as mock_update:
                metrics = {"win_rate": 0.75, "total_pnl": 100.0}
                extra = {"component": "backtester"}

                self.logger.performance("Performance update", metrics, extra=extra)

                # Verify log was called with correct level
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[0][0] == PERF_LEVEL  # 22
                assert call_args[0][1] == "Performance update"
                assert call_args[1]['extra'] == extra

                # Verify _update_performance was called
                mock_update.assert_called_once_with(metrics)

    def test_performance_method_handles_type_error(self):
        """Test performance method handles TypeError gracefully."""
        # Mock the log method to simulate old Python logging behavior
        original_log = self.logger.log
        def mock_log(level, msg, *args, **kwargs):
            if 'extra' in kwargs:
                raise TypeError("Old Python logging")
            return original_log(level, msg, *args, **kwargs)

        with patch.object(self.logger, 'log', mock_log):
            with patch.object(self.logger, '_update_performance') as mock_update:
                metrics = {"win_rate": 0.75}

                # Should not raise exception
                self.logger.performance("Performance update", metrics)

                # Should still call _update_performance
                mock_update.assert_called_once_with(metrics)

    def test_log_signal_with_dict(self):
        """Test log_signal with dictionary input (lines 187-192)."""
        with patch.object(self.logger, 'trade') as mock_trade:
            signal = {"symbol": "BTC/USDT", "action": "BUY", "price": 50000}

            self.logger.log_signal(signal, extra={"component": "strategy"})

            mock_trade.assert_called_once()
            call_args = mock_trade.call_args
            assert call_args[0][0] == "New trading signal"
            assert call_args[0][1] == {"signal": signal}
            assert call_args[1]['extra'] == {"component": "strategy"}

    def test_log_signal_with_object(self):
        """Test log_signal with object input."""
        with patch.object(self.logger, 'trade') as mock_trade:
            # Mock signal object
            signal_obj = MagicMock()
            signal_obj.__dict__ = {"symbol": "BTC/USDT", "action": "BUY"}

            with patch('utils.logger.signal_to_dict', return_value={"symbol": "BTC/USDT", "action": "BUY"}):
                self.logger.log_signal(signal_obj)

                mock_trade.assert_called_once()

    def test_log_signal_handles_conversion_error(self):
        """Test log_signal handles conversion errors."""
        with patch('utils.logger.signal_to_dict', side_effect=TypeError("Conversion failed")):
            signal_obj = MagicMock()

            with pytest.raises(TypeError):
                self.logger.log_signal(signal_obj)

    def test_log_order_with_dict(self):
        """Test log_order with dictionary input (lines 199-204)."""
        with patch.object(self.logger, 'trade') as mock_trade:
            order = {"id": "12345", "symbol": "BTC/USDT", "side": "BUY"}
            extra = {"correlation_id": "abc123"}

            self.logger.log_order(order, "live", extra=extra)

            mock_trade.assert_called_once()
            call_args = mock_trade.call_args
            expected_order_data = {"id": "12345", "symbol": "BTC/USDT", "side": "BUY", "mode": "live"}
            assert call_args[0][1] == expected_order_data
            assert call_args[1]['extra'] == extra

    def test_log_order_with_non_dict(self):
        """Test log_order with non-dictionary input."""
        with patch.object(self.logger, 'trade') as mock_trade:
            order = "order_string"

            self.logger.log_order(order, "paper")

            mock_trade.assert_called_once()
            call_args = mock_trade.call_args
            assert call_args[0][1] == {"order": "order_string", "mode": "paper"}

    def test_log_order_handles_error(self):
        """Test log_order handles errors gracefully."""
        with patch.object(self.logger, 'trade', side_effect=TypeError("Trade logging failed")):
            order = {"id": "12345"}

            with pytest.raises(TypeError):
                self.logger.log_order(order, "live")

    def test_log_rejected_signal(self):
        """Test log_rejected_signal method."""
        with patch.object(self.logger, 'trade') as mock_trade:
            signal = {"symbol": "BTC/USDT"}
            reason = "Insufficient balance"

            self.logger.log_rejected_signal(signal, reason)

            mock_trade.assert_called_once()
            call_args = mock_trade.call_args
            assert "Signal rejected: Insufficient balance" in call_args[0][0]
            assert call_args[0][1]["signal"] == signal
            assert call_args[0][1]["reason"] == reason

    def test_log_failed_order(self):
        """Test log_failed_order method."""
        with patch.object(self.logger, 'trade') as mock_trade:
            signal = {"symbol": "BTC/USDT"}
            error = "Exchange timeout"

            self.logger.log_failed_order(signal, error)

            mock_trade.assert_called_once()
            call_args = mock_trade.call_args
            assert "Order failed: Exchange timeout" in call_args[0][0]
            assert call_args[0][1]["signal"] == signal
            assert call_args[0][1]["error"] == error

    def test_record_trade_updates_stats(self):
        """Test _record_trade updates performance statistics (lines 235-236)."""
        initial_stats = self.logger.performance_stats.copy()

        trade_data = {
            "pair": "BTC/USDT",
            "action": "BUY",
            "pnl": 100.0,
            "size": 1.0,
            "entry_price": 50000,
            "exit_price": 51000
        }

        self.logger._record_trade(trade_data)

        # Verify trade was recorded
        assert len(self.logger.trades) == 1
        assert self.logger.trades[0] == trade_data

        # Verify stats were updated
        assert self.logger.performance_stats["total_trades"] == initial_stats["total_trades"] + 1
        assert self.logger.performance_stats["wins"] == initial_stats["wins"] + 1
        assert self.logger.performance_stats["total_pnl"] == initial_stats["total_pnl"] + 100.0
        assert self.logger.performance_stats["max_win"] == 100.0

    def test_record_trade_loss_updates_stats(self):
        """Test _record_trade updates stats for losing trades."""
        trade_data = {
            "pair": "BTC/USDT",
            "action": "SELL",
            "pnl": -50.0
        }

        self.logger._record_trade(trade_data)

        assert self.logger.performance_stats["losses"] == 1
        assert self.logger.performance_stats["total_pnl"] == -50.0
        assert self.logger.performance_stats["max_loss"] == -50.0

    def test_record_trade_csv_persistence(self):
        """Test _record_trade persists to CSV."""
        trade_data = {
            "pair": "BTC/USDT",
            "action": "BUY",
            "pnl": 100.0,
            "size": 1.0,
            "entry_price": 50000,
            "exit_price": 51000,
            "timestamp": "2023-01-01T00:00:00Z"
        }

        self.logger._record_trade(trade_data)

        # Verify CSV was written
        assert self.logger.trade_csv.exists()

        with open(self.logger.trade_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header + 1 data row
        assert len(rows) == 2
        data_row = rows[1]
        assert data_row[1] == "BTC/USDT"  # pair
        assert data_row[2] == "BUY"  # action
        assert data_row[6] == "100.0"  # pnl

    def test_record_trade_handles_csv_io_error(self):
        """Test _record_trade handles CSV I/O errors gracefully."""
        with patch('builtins.open', side_effect=OSError("Disk full")):
            trade_data = {"pair": "BTC/USDT", "pnl": 100.0}

            # Should not raise exception
            self.logger._record_trade(trade_data)

    def test_record_trade_handles_data_error(self):
        """Test _record_trade handles data formatting errors."""
        # Invalid pnl value
        trade_data = {"pnl": "invalid"}

        with pytest.raises((TypeError, ValueError)):
            self.logger._record_trade(trade_data)

    def test_update_performance_numeric_metrics(self):
        """Test _update_performance with numeric metrics (lines 267-281)."""
        initial_pnl = self.logger.performance_stats.get("custom_metric", 0.0)

        metrics = {"custom_metric": 25.5, "another_metric": 10}

        self.logger._update_performance(metrics)

        assert self.logger.performance_stats["custom_metric"] == initial_pnl + 25.5
        assert self.logger.performance_stats["another_metric"] == 10.0

    def test_update_performance_non_numeric_ignored(self):
        """Test _update_performance ignores non-numeric values."""
        metrics = {"string_metric": "ignored", "numeric_metric": 42.0}

        self.logger._update_performance(metrics)

        assert "string_metric" not in self.logger.performance_stats
        assert self.logger.performance_stats["numeric_metric"] == 42.0

    def test_update_performance_handles_type_error(self):
        """Test _update_performance handles type conversion errors."""
        # Test with a value that will cause float() to fail
        metrics = {"bad_metric": None}

        # This should not raise an exception - None values are ignored
        self.logger._update_performance(metrics)

        # Verify the bad_metric was not added
        assert "bad_metric" not in self.logger.performance_stats

    def test_display_performance_returns_copy(self):
        """Test display_performance returns a copy (lines 291-296)."""
        self.logger.performance_stats["test_key"] = "test_value"

        result = self.logger.display_performance()

        assert result == self.logger.performance_stats
        assert result is not self.logger.performance_stats  # Should be a copy

    def test_get_trade_history_with_limit(self):
        """Test get_trade_history with limit (line 300)."""
        # Add some trades
        for i in range(5):
            self.logger.trades.append({"id": i, "pnl": i * 10})

        # Get last 3 trades
        history = self.logger.get_trade_history(limit=3)

        assert len(history) == 3
        # Should return most recent first
        assert history[0]["id"] == 4
        assert history[1]["id"] == 3
        assert history[2]["id"] == 2

    def test_get_trade_history_no_limit(self):
        """Test get_trade_history without limit."""
        for i in range(3):
            self.logger.trades.append({"id": i})

        history = self.logger.get_trade_history()

        assert len(history) == 3

    def test_get_trade_history_empty(self):
        """Test get_trade_history with empty trades."""
        history = self.logger.get_trade_history()

        assert history == []

    def test_get_trade_history_invalid_limit(self):
        """Test get_trade_history with invalid limit."""
        with pytest.raises((TypeError, IndexError)):
            self.logger.get_trade_history(limit="invalid")

    def test_get_performance_stats_wrapper(self):
        """Test get_performance_stats wrapper (lines 307-317)."""
        with patch.object(self.logger, 'display_performance', return_value={"test": "data"}) as mock_display:
            result = self.logger.get_performance_stats()

            mock_display.assert_called_once()
            assert result == {"test": "data"}


class TestSetupLogging:
    """Test cases for setup_logging function."""

    def teardown_method(self):
        """Clean up after each test."""
        # Reset global logger
        import utils.logger
        utils.logger._GLOBAL_TRADE_LOGGER = None

        # Clear any existing handlers
        logger = logging.getLogger("crypto_bot")
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    def test_setup_logging_default_config(self):
        """Test setup_logging with default configuration (lines 321-325)."""
        logger = setup_logging()

        assert isinstance(logger, TradeLogger)
        assert logger.name == "crypto_bot"
        assert logger.level == logging.INFO

        # Should have console handler
        assert len(logger.handlers) >= 1

    def test_setup_logging_custom_config(self):
        """Test setup_logging with custom configuration."""
        config = {
            "level": "DEBUG",
            "file_logging": False,
            "console": True,
            "log_file": "custom.log",
            "max_size": 1024,
            "backup_count": 2
        }

        logger = setup_logging(config)

        assert logger.level == logging.DEBUG
        # Should have at least console handler
        assert len(logger.handlers) >= 1

    def test_setup_logging_creates_file_handler(self):
        """Test setup_logging creates rotating file handler."""
        import tempfile
        import os

        # Ensure not in test mode
        os.environ["TESTING"] = "0"

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp_file:
            log_file = temp_file.name

        try:
            config = {
                "file_logging": True,
                "log_file": log_file,
                "max_size": 1024,
                "backup_count": 1
            }

            logger = setup_logging(config)

            # Should have file handler
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
            assert len(file_handlers) == 1

            file_handler = file_handlers[0]
            assert file_handler.baseFilename == log_file
            assert file_handler.maxBytes == 1024
            assert file_handler.backupCount == 1
        finally:
            try:
                os.unlink(log_file)
            except:
                pass

    def test_setup_logging_no_console_handler(self):
        """Test setup_logging without console handler."""
        config = {"console": False, "file_logging": False}

        logger = setup_logging(config)

        # Should have no handlers
        assert len(logger.handlers) == 0

    def test_setup_logging_reuses_global_logger(self):
        """Test setup_logging reuses existing global logger."""
        logger1 = setup_logging()
        logger2 = setup_logging()

        assert logger1 is logger2


class TestGlobalFunctions:
    """Test cases for global utility functions."""

    def teardown_method(self):
        """Clean up after each test."""
        # Reset global logger
        import utils.logger
        utils.logger._GLOBAL_TRADE_LOGGER = None

    def test_get_trade_logger_creates_instance(self):
        """Test get_trade_logger creates instance when none exists (line 431)."""
        logger = get_trade_logger()

        assert isinstance(logger, TradeLogger)
        assert logger.name == "crypto_bot"

    def test_get_trade_logger_returns_existing(self):
        """Test get_trade_logger returns existing instance."""
        logger1 = get_trade_logger()
        logger2 = get_trade_logger()

        assert logger1 is logger2

    def test_generate_correlation_id_format(self):
        """Test generate_correlation_id returns valid UUID hex (lines 442-452)."""
        correlation_id = generate_correlation_id()

        assert isinstance(correlation_id, str)
        assert len(correlation_id) == 32  # UUID hex length

        # Should be valid hex
        int(correlation_id, 16)

    def test_generate_correlation_id_uniqueness(self):
        """Test generate_correlation_id generates unique IDs."""
        ids = {generate_correlation_id() for _ in range(100)}

        assert len(ids) == 100  # All should be unique

    def test_get_logger_with_context(self):
        """Test get_logger_with_context creates LoggerAdapter."""
        adapter = get_logger_with_context(
            symbol="BTC/USDT",
            component="order_manager",
            correlation_id="test123"
        )

        assert isinstance(adapter, logging.LoggerAdapter)
        assert adapter.extra == {
            "symbol": "BTC/USDT",
            "component": "order_manager",
            "correlation_id": "test123"
        }

    def test_get_logger_with_context_partial_args(self):
        """Test get_logger_with_context with partial arguments."""
        adapter = get_logger_with_context(symbol="BTC/USDT")

        assert adapter.extra == {"symbol": "BTC/USDT"}

    def test_get_logger_with_context_no_args(self):
        """Test get_logger_with_context with no arguments."""
        adapter = get_logger_with_context()

        assert adapter.extra == {}


class TestLogToFile:
    """Test cases for log_to_file function."""

    def test_log_to_file_creates_new_file(self):
        """Test log_to_file creates new JSON file (lines 463-479)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.json")

            data = {"key": "value", "number": 42}

            log_to_file(data, os.path.join(temp_dir, "test"))

            assert os.path.exists(test_file)

            with open(test_file, 'r') as f:
                content = json.load(f)

            assert content == data

    def test_log_to_file_appends_to_existing(self):
        """Test log_to_file appends to existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.json")

            # Create initial file
            initial_data = {"first": "entry"}
            with open(test_file, 'w') as f:
                json.dump(initial_data, f)

            # Append new data
            new_data = {"second": "entry"}
            log_to_file(new_data, os.path.join(temp_dir, "test"))

            # File should contain both entries
            with open(test_file, 'r') as f:
                content = f.read()

            # The content should contain both JSON objects
            # The exact format depends on how json.dump formats it
            assert '"first": "entry"' in content
            assert '"second": "entry"' in content

    def test_log_to_file_handles_io_error(self):
        """Test log_to_file handles I/O errors gracefully."""
        with patch('builtins.open', side_effect=OSError("Permission denied")):
            data = {"test": "data"}

            # Should not raise exception for I/O errors
            log_to_file(data, "test")

    def test_log_to_file_handles_serialization_with_default(self):
        """Test log_to_file handles non-serializable objects using default converter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use a lambda function which cannot be serialized by JSON
            data = {"bad_data": lambda x: x}

            # Should not raise exception - uses default=str converter
            log_to_file(data, os.path.join(temp_dir, "test"))

            # File should be created
            test_file = os.path.join(temp_dir, "test.json")
            assert os.path.exists(test_file)

            # Verify the lambda was converted to string
            with open(test_file, 'r') as f:
                content = json.load(f)

            assert "bad_data" in content
            # Lambda should be converted to string representation
            assert isinstance(content["bad_data"], str)

    def test_log_to_file_with_datetime(self):
        """Test log_to_file handles datetime objects."""
        data = {"timestamp": datetime.now()}

        with tempfile.TemporaryDirectory() as temp_dir:
            log_to_file(data, os.path.join(temp_dir, "test"))

            # Should not raise exception and should serialize datetime
            test_file = os.path.join(temp_dir, "test.json")
            assert os.path.exists(test_file)


class TestJSONFormatter:
    """Test cases for JSONFormatter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = JSONFormatter()

    def test_format_basic_record(self):
        """Test formatting basic log record as JSON."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = self.formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["timestamp"].endswith("Z")
        assert parsed["level"] == "INFO"
        assert parsed["module"] == "test_logger"
        assert parsed["message"] == "Test message"
        assert parsed["correlation_id"] is None
        assert parsed["request_id"] is None
        assert parsed["strategy_id"] is None

    def test_format_record_with_context(self):
        """Test formatting log record with context fields."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message with context",
            args=(),
            exc_info=None
        )

        # Add context fields
        record.correlation_id = "corr_123"
        record.request_id = "req_456"
        record.strategy_id = "momentum_v1"
        record.component = "order_manager"
        record.symbol = "BTC/USDT"

        formatted = self.formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["correlation_id"] == "corr_123"
        assert parsed["request_id"] == "req_456"
        assert parsed["strategy_id"] == "momentum_v1"
        assert parsed["component"] == "order_manager"
        assert parsed["symbol"] == "BTC/USDT"

    def test_format_record_with_exception(self):
        """Test formatting log record with exception info."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        formatted = self.formatter.format(record)
        parsed = json.loads(formatted)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "Test exception" in parsed["exception"]


class TestPrettyFormatter:
    """Test cases for PrettyFormatter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = PrettyFormatter()

    def test_format_basic_record(self):
        """Test formatting basic log record with pretty format."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = self.formatter.format(record)

        assert "Test message" in formatted
        assert "[INFO]" in formatted
        assert "test_logger:" in formatted

    def test_format_record_with_context(self):
        """Test formatting log record with context fields."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        # Add context fields
        record.correlation_id = "corr_123"
        record.request_id = "req_456"
        record.strategy_id = "momentum_v1"
        record.component = "order_manager"

        formatted = self.formatter.format(record)

        assert "Test message" in formatted
        assert "corr_id=corr_123" in formatted
        assert "req_id=req_456" in formatted
        assert "strategy=momentum_v1" in formatted
        assert "component=order_manager" in formatted


class TestEnvironmentVariables:
    """Test cases for environment variable support."""

    def teardown_method(self):
        """Clean up after each test."""
        # Reset global logger
        import utils.logger
        utils.logger._GLOBAL_TRADE_LOGGER = None

        # Clear environment variables
        for var in ["LOG_LEVEL", "LOG_FILE", "LOG_FORMAT", "TESTING"]:
            os.environ.pop(var, None)

    def test_setup_logging_with_env_log_level(self):
        """Test setup_logging uses LOG_LEVEL environment variable."""
        os.environ["LOG_LEVEL"] = "DEBUG"

        logger = setup_logging()

        assert logger.level == logging.DEBUG

    def test_setup_logging_with_invalid_env_log_level(self):
        """Test setup_logging handles invalid LOG_LEVEL gracefully."""
        os.environ["LOG_LEVEL"] = "INVALID"

        with patch('builtins.print') as mock_print:
            logger = setup_logging()

            # Should default to INFO
            assert logger.level == logging.INFO
            mock_print.assert_called_once()

    def test_setup_logging_with_env_log_file(self):
        """Test setup_logging uses LOG_FILE environment variable."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as temp_file:
            log_file = temp_file.name

        try:
            os.environ["LOG_FILE"] = log_file

            logger = setup_logging()

            # Should have file handler with correct path
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
            assert len(file_handlers) == 1
            assert file_handlers[0].baseFilename == log_file
        finally:
            try:
                os.unlink(log_file)
            except:
                pass

    def test_setup_logging_with_env_log_format_json(self):
        """Test setup_logging uses LOG_FORMAT=json environment variable."""
        os.environ["LOG_FORMAT"] = "json"

        logger = setup_logging()

        # Console handler should use JSONFormatter
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(console_handlers) == 1
        assert isinstance(console_handlers[0].formatter, JSONFormatter)

    def test_setup_logging_with_env_log_format_pretty(self):
        """Test setup_logging uses LOG_FORMAT=pretty environment variable."""
        os.environ["LOG_FORMAT"] = "pretty"

        logger = setup_logging()

        # Console handler should use PrettyFormatter
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) == 1
        assert isinstance(console_handlers[0].formatter, PrettyFormatter)

    def test_setup_logging_with_env_log_format_color(self):
        """Test setup_logging uses LOG_FORMAT=color environment variable."""
        os.environ["LOG_FORMAT"] = "color"

        logger = setup_logging()

        # Console handler should use ColorFormatter
        console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) == 1
        assert isinstance(console_handlers[0].formatter, ColorFormatter)


class TestCorrelationIdSupport:
    """Test cases for correlation_id and request_id support."""

    def test_generate_request_id_format(self):
        """Test generate_request_id returns properly formatted ID."""
        request_id = generate_request_id()

        assert isinstance(request_id, str)
        assert request_id.startswith("req_")
        assert len(request_id) == 21  # "req_" + 16 hex chars

    def test_generate_request_id_uniqueness(self):
        """Test generate_request_id generates unique IDs."""
        ids = {generate_request_id() for _ in range(100)}

        assert len(ids) == 100  # All should be unique

    def test_get_logger_with_context_full_args(self):
        """Test get_logger_with_context with all arguments."""
        adapter = get_logger_with_context(
            symbol="BTC/USDT",
            component="order_manager",
            correlation_id="corr_123",
            request_id="req_456",
            strategy_id="momentum_v1"
        )

        expected_extra = {
            "symbol": "BTC/USDT",
            "component": "order_manager",
            "correlation_id": "corr_123",
            "request_id": "req_456",
            "strategy_id": "momentum_v1"
        }

        assert adapter.extra == expected_extra

    def test_get_logger_with_context_partial_args(self):
        """Test get_logger_with_context with partial arguments."""
        adapter = get_logger_with_context(
            correlation_id="corr_123",
            request_id="req_456"
        )

        expected_extra = {
            "correlation_id": "corr_123",
            "request_id": "req_456"
        }

        assert adapter.extra == expected_extra

    def test_logger_adapter_preserves_context(self):
        """Test that LoggerAdapter preserves context across log calls."""
        adapter = get_logger_with_context(
            symbol="BTC/USDT",
            correlation_id="corr_123"
        )

        with patch.object(adapter.logger, 'log') as mock_log:
            adapter.info("First message")
            adapter.warning("Second message")

            # Both calls should have the same context
            first_call = mock_log.call_args_list[0]
            second_call = mock_log.call_args_list[1]

            assert first_call[1]['extra']['symbol'] == "BTC/USDT"
            assert first_call[1]['extra']['correlation_id'] == "corr_123"
            assert second_call[1]['extra']['symbol'] == "BTC/USDT"
            assert second_call[1]['extra']['correlation_id'] == "corr_123"


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.logger = TradeLogger("integration_test")

    def teardown_method(self):
        """Clean up after each test."""
        # Clean up log files
        if hasattr(self.logger, 'trade_csv') and self.logger.trade_csv.exists():
            try:
                self.logger.trade_csv.unlink()
            except:
                pass

    def test_complete_trade_workflow(self):
        """Test complete trade logging workflow."""
        # Log a trade
        trade_data = {
            "pair": "BTC/USDT",
            "action": "BUY",
            "size": 1.0,
            "entry_price": 50000,
            "exit_price": 51000,
            "pnl": 1000.0
        }

        self.logger.trade("Executed trade", trade_data)

        # Verify trade was recorded
        assert len(self.logger.trades) == 1

        # Verify performance stats
        stats = self.logger.display_performance()
        assert stats["total_trades"] == 1
        assert stats["wins"] == 1
        assert stats["total_pnl"] == 1000.0

        # Verify CSV persistence
        assert self.logger.trade_csv.exists()

    def test_performance_tracking_workflow(self):
        """Test performance tracking workflow."""
        # Log multiple trades
        trades = [
            {"pnl": 100.0},
            {"pnl": -50.0},
            {"pnl": 200.0}
        ]

        for trade in trades:
            self.logger._record_trade(trade)

        # Log performance metrics
        metrics = {"custom_metric": 75.5}
        self.logger.performance("Performance update", metrics)

        # Verify combined stats
        stats = self.logger.display_performance()
        assert stats["total_trades"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["total_pnl"] == 250.0
        assert stats["custom_metric"] == 75.5

    def test_error_handling_workflow(self):
        """Test error handling in logging workflow."""
        # Test with invalid trade data
        with pytest.raises((TypeError, ValueError)):
            self.logger._record_trade({"pnl": "invalid"})

        # Logger should still be functional after error
        # Clear any trades that might have been added
        self.logger.trades.clear()

        # Now test that it works with valid data
        self.logger._record_trade({"pnl": 100.0})
        assert len(self.logger.trades) == 1

    def test_context_logging_workflow(self):
        """Test context-aware logging workflow."""
        # Create logger with context
        adapter = get_logger_with_context(
            symbol="BTC/USDT",
            component="test_component",
            correlation_id="test123"
        )

        # Log with context
        with patch.object(adapter.logger, 'log') as mock_log:
            adapter.info("Test message")

            # Verify context was included
            call_args = mock_log.call_args
            assert call_args[1]['extra']['symbol'] == "BTC/USDT"
            assert call_args[1]['extra']['component'] == "test_component"
            assert call_args[1]['extra']['correlation_id'] == "test123"

    def test_structured_logging_integration(self):
        """Test structured logging with JSON output."""
        # Create logger with JSON format
        config = {"format": "json", "console": True, "file_logging": False}
        logger = setup_logging(config)

        # Create adapter with context
        adapter = get_logger_with_context(
            symbol="BTC/USDT",
            component="order_executor",
            correlation_id="corr_123",
            request_id="req_456",
            strategy_id="momentum_v1"
        )

        # Capture console output
        import io
        from contextlib import redirect_stdout

        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            adapter.info("Order execution started")

        output = captured_output.getvalue()

        # Parse JSON output
        parsed = json.loads(output.strip())

        # Verify structured fields
        assert parsed["level"] == "INFO"
        assert parsed["module"] == "crypto_bot"
        assert parsed["message"] == "Order execution started"
        assert parsed["correlation_id"] == "corr_123"
        assert parsed["request_id"] == "req_456"
        assert parsed["strategy_id"] == "momentum_v1"
        assert parsed["symbol"] == "BTC/USDT"
        assert parsed["component"] == "order_executor"
        assert "timestamp" in parsed
