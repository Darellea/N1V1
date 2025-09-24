"""
Comprehensive Signal Router Testing Suite
===========================================

This module provides comprehensive testing for the Signal Router facade and its components,
covering core functionality, state management, integration, and edge cases.
"""

import asyncio
import time
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import json
import os
from pathlib import Path
from decimal import Decimal
import pandas as pd

from core.signal_router import SignalRouter, JournalWriter
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import OrderType
from utils.adapter import signal_to_dict
from utils.time import now_ms


class TestSignalRouterFacade:
    """Test the signal_router facade imports and re-exports."""

    def test_imports(self):
        """Test that SignalRouter and JournalWriter can be imported from facade."""
        # This covers the import lines in core/signal_router.py
        from core.signal_router import SignalRouter, JournalWriter
        assert SignalRouter is not None
        assert JournalWriter is not None

    def test_re_exports(self):
        """Test that re-exports match the actual classes."""
        from core.signal_router.router import SignalRouter as RealSignalRouter
        from core.signal_router.router import JournalWriter as RealJournalWriter
        from core.signal_router import SignalRouter as FacadeSignalRouter
        from core.signal_router import JournalWriter as FacadeJournalWriter

        assert FacadeSignalRouter is RealSignalRouter
        assert FacadeJournalWriter is RealJournalWriter


class TestJournalWriter:
    """Test JournalWriter functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.journal_path = Path(self.temp_dir) / "test_journal.jsonl"
        self.task_manager = MagicMock()

    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.journal_path.exists():
            self.journal_path.unlink()
        os.rmdir(self.temp_dir)

    def test_initialization(self):
        """Test JournalWriter initialization."""
        jw = JournalWriter(self.journal_path, self.task_manager)
        assert jw.path == self.journal_path
        assert jw.task_manager == self.task_manager
        assert jw._task is None

    def test_append_synchronous_fallback(self):
        """Test synchronous append when no event loop."""
        jw = JournalWriter(self.journal_path)

        entry = {"test": "data", "timestamp": now_ms()}
        jw.append(entry)

        # Verify entry was written
        assert self.journal_path.exists()
        with open(self.journal_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["test"] == "data"

    @pytest.mark.asyncio
    async def test_append_with_event_loop(self):
        """Test append with running event loop."""
        jw = JournalWriter(self.journal_path, self.task_manager)

        # Mock task manager
        self.task_manager.create_task = MagicMock()

        entry = {"test": "async_data", "timestamp": now_ms()}
        jw.append(entry)

        # Should have created task
        self.task_manager.create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_method(self):
        """Test stopping the journal writer."""
        jw = JournalWriter(self.journal_path, self.task_manager)

        # Mock task
        mock_task = AsyncMock()
        jw._task = mock_task

        await jw.stop()

        # Should have put None in queue and waited for task
        mock_task.assert_awaited_once()

    def test_multiple_appends(self):
        """Test multiple synchronous appends."""
        jw = JournalWriter(self.journal_path)

        entries = [
            {"id": 1, "action": "store"},
            {"id": 2, "action": "cancel"},
            {"id": 3, "action": "store"}
        ]

        for entry in entries:
            jw.append(entry)

        # Verify all entries written
        with open(self.journal_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3
            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["id"] == i + 1


class TestSignalRouterCore:
    """Test core SignalRouter functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = MagicMock()
        self.risk_manager.evaluate_signal = AsyncMock(return_value=True)
        self.risk_manager.validate_signal = MagicMock(return_value=True)
        self.task_manager = MagicMock()

        # Mock get_config
        self.config_patcher = patch('core.signal_router.router.get_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False}
        }

    def teardown_method(self):
        """Cleanup patches."""
        self.config_patcher.stop()

    def test_initialization(self):
        """Test SignalRouter initialization."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        assert router.risk_manager == self.risk_manager
        assert router.task_manager == self.task_manager
        assert len(router.active_signals) == 0
        assert len(router.signal_history) == 0
        assert not router.block_signals
        assert router.critical_errors == 0

    @pytest.mark.asyncio
    async def test_process_signal_basic(self):
        """Test basic signal processing."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.MODERATE,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )

        result = await router.process_signal(signal)

        assert result is not None
        assert result.symbol == "BTC/USD"
        self.risk_manager.evaluate_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_signal_risk_rejection(self):
        """Test signal rejection by risk manager."""
        self.risk_manager.evaluate_signal = AsyncMock(return_value=False)

        router = SignalRouter(self.risk_manager, self.task_manager)

        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.MODERATE,
            order_type="market",
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )

        result = await router.process_signal(signal)

        assert result is None
        self.risk_manager.evaluate_signal.assert_called_once()

    def test_validate_signal(self):
        """Test signal validation."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Valid signal
        valid_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )
        assert router._validate_signal(valid_signal)

        # Invalid signal - no symbol
        invalid_signal = TradingSignal(
            strategy_id="test",
            symbol="",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )
        assert not router._validate_signal(invalid_signal)

        # Invalid signal - zero amount
        invalid_signal2 = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("0"),
            timestamp=int(time.time())
        )
        assert not router._validate_signal(invalid_signal2)

    def test_validate_timestamp(self):
        """Test timestamp validation."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Valid timestamp
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )
        assert router._validate_timestamp(signal)

        # Invalid timestamp - too old
        signal_invalid = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=0
        )
        assert not router._validate_timestamp(signal_invalid)

    def test_get_active_signals(self):
        """Test getting active signals."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Empty initially
        assert router.get_active_signals() == []

        # Add a signal
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )
        signal_id = router._generate_signal_id(signal)
        router.active_signals[signal_id] = signal

        # Get all signals
        active = router.get_active_signals()
        assert len(active) == 1
        assert active[0].symbol == "BTC/USD"

        # Get by symbol
        btc_signals = router.get_active_signals("BTC/USD")
        assert len(btc_signals) == 1

        eth_signals = router.get_active_signals("ETH/USD")
        assert len(eth_signals) == 0

    def test_get_signal_history(self):
        """Test getting signal history."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Empty initially
        assert router.get_signal_history() == []

        # Add to history
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )
        router.signal_history.append(signal)

        history = router.get_signal_history()
        assert len(history) == 1

        # Test limit
        history_limited = router.get_signal_history(0)
        assert len(history_limited) == 0

    def test_clear_signals(self):
        """Test clearing all signals."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Add a signal
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )
        signal_id = router._generate_signal_id(signal)
        router.active_signals[signal_id] = signal
        router.signal_history.append(signal)

        router.clear_signals()

        assert len(router.active_signals) == 0
        assert len(router.signal_history) == 1  # History not cleared

    @pytest.mark.asyncio
    async def test_update_signal_status(self):
        """Test updating signal status."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )
        signal_id = router._generate_signal_id(signal)
        router.active_signals[signal_id] = signal

        await router.update_signal_status(signal, "executed", "test reason")

        # Signal should be removed from active
        assert signal_id not in router.active_signals

    def test_generate_signal_id(self):
        """Test signal ID generation."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Use a reasonable timestamp in seconds
        timestamp = 1234567890
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=timestamp
        )

        signal_id = router._generate_signal_id(signal)
        # The timestamp is converted to datetime, so it will be a datetime string
        import datetime
        expected_timestamp = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        expected = f"test_strategy_BTC/USD_{expected_timestamp}"
        assert signal_id == expected


class TestSignalRouterConflictResolution:
    """Test signal conflict resolution."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = MagicMock()
        self.risk_manager.evaluate_signal = AsyncMock(return_value=True)
        self.risk_manager.validate_signal = MagicMock(return_value=True)
        self.risk_manager.require_stop_loss = False
        self.task_manager = MagicMock()

        self.config_patcher = patch('core.signal_router.router.get_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False}
        }

    def teardown_method(self):
        """Cleanup patches."""
        self.config_patcher.stop()

    def test_check_signal_conflicts_no_conflicts(self):
        """Test checking conflicts when none exist."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )

        conflicts = router._check_signal_conflicts(signal)
        assert len(conflicts) == 0

    def test_check_signal_conflicts_opposite_signals(self):
        """Test detecting opposite signal conflicts."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Add existing long entry
        existing_signal = TradingSignal(
            strategy_id="existing",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )
        existing_id = router._generate_signal_id(existing_signal)
        router.active_signals[existing_id] = existing_signal

        # New short entry should conflict
        new_signal = TradingSignal(
            strategy_id="new",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time()) + 1
        )

        conflicts = router._check_signal_conflicts(new_signal)
        assert len(conflicts) == 1
        assert conflicts[0].strategy_id == "existing"

    def test_is_opposite_signal(self):
        """Test opposite signal detection."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        long_entry = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )

        short_entry = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )

        long_exit = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.EXIT_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )

        # Entry signals should be opposite
        assert router._is_opposite_signal(long_entry, short_entry)
        assert router._is_opposite_signal(short_entry, long_entry)

        # Entry and exit should be opposite
        assert router._is_opposite_signal(long_entry, long_exit)

        # Same direction should not be opposite
        assert not router._is_opposite_signal(long_entry, long_entry)


class TestSignalRouterIntegration:
    """Test SignalRouter integration aspects."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = MagicMock()
        self.risk_manager.evaluate_signal = AsyncMock(return_value=True)
        self.risk_manager.validate_signal = MagicMock(return_value=True)
        self.risk_manager.require_stop_loss = False
        self.task_manager = MagicMock()

        self.config_patcher = patch('core.signal_router.router.get_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False}
        }

    def teardown_method(self):
        """Cleanup patches."""
        self.config_patcher.stop()

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism for risk manager calls."""
        # Make risk manager fail twice then succeed
        self.risk_manager.evaluate_signal = AsyncMock(side_effect=[Exception("Fail1"), Exception("Fail2"), True])

        router = SignalRouter(self.risk_manager, self.task_manager)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )

        result = await router.process_signal(signal)

        # Should have been called 3 times (initial + 2 retries)
        assert self.risk_manager.evaluate_signal.call_count == 3
        assert result is not None

    @pytest.mark.asyncio
    async def test_critical_error_blocking(self):
        """Test blocking signals after critical errors."""
        self.risk_manager.evaluate_signal = AsyncMock(side_effect=Exception("Critical failure"))

        router = SignalRouter(self.risk_manager, self.task_manager, safe_mode_threshold=2)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )

        # First failure
        result1 = await router.process_signal(signal)
        assert result1 is None
        assert router.critical_errors == 1
        assert not router.block_signals

        # Second failure should trigger blocking
        result2 = await router.process_signal(signal)
        assert result2 is None
        assert router.critical_errors == 2
        assert router.block_signals

        # Third attempt should be blocked immediately
        result3 = await router.process_signal(signal)
        assert result3 is None
        # Should not call risk manager again (each failed call retries 3 times: initial + 2 retries)
        assert self.risk_manager.evaluate_signal.call_count == 6


class TestSignalRouterEdgeCases:
    """Test SignalRouter edge cases."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = MagicMock()
        self.risk_manager.evaluate_signal = AsyncMock(return_value=True)
        self.risk_manager.validate_signal = MagicMock(return_value=True)
        self.task_manager = MagicMock()

        self.config_patcher = patch('core.signal_router.router.get_config')
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False}
        }

    def teardown_method(self):
        """Cleanup patches."""
        self.config_patcher.stop()

    def test_signal_with_quantity_instead_of_amount(self):
        """Test backward compatibility with quantity field."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Create signal with quantity instead of amount
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=None,
            quantity=Decimal("1.0"),  # Use deprecated field
            stop_loss=Decimal("49000"),
            timestamp=int(time.time())
        )

        assert router._validate_signal(signal)

    def test_signal_without_stop_loss_for_entry(self):
        """Test signal validation requiring stop loss."""
        # Mock risk manager to require stop loss
        self.risk_manager.require_stop_loss = True

        router = SignalRouter(self.risk_manager, self.task_manager)

        # Signal without stop loss
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )

        assert not router._validate_signal(signal)

        # Signal with stop loss
        signal.stop_loss = Decimal("49000")
        assert router._validate_signal(signal)

    def test_invalid_timestamp_formats(self):
        """Test handling of various invalid timestamp formats."""
        router = SignalRouter(self.risk_manager, self.task_manager)

        # Test with negative timestamp
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time())
        )
        signal._original_timestamp = -1000  # Modify the original timestamp
        assert not router._validate_timestamp(signal)

        # Test with future timestamp (too far)
        signal._original_timestamp = int(time.time() + 157784630)  # 5+ years in future
        assert not router._validate_timestamp(signal)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
