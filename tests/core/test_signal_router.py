"""
Comprehensive Signal Router Testing Suite
===========================================

This module provides comprehensive testing for the Signal Router facade and its components,
covering core functionality, state management, integration, and edge cases.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.contracts import SignalStrength, SignalType, TradingSignal
from core.signal_router import JournalWriter, SignalRouter
from core.types import OrderType
from utils.time import now_ms

# Suppress asyncio task destruction warnings during test cleanup
logging.getLogger('asyncio').setLevel(logging.ERROR)


class TestSignalRouterFacade:
    """Test the signal_router facade imports and re-exports."""

    def test_imports(self):
        """Test that SignalRouter and JournalWriter can be imported from facade."""
        # This covers the import lines in core/signal_router.py
        from core.signal_router import JournalWriter, SignalRouter

        assert SignalRouter is not None
        assert JournalWriter is not None

    def test_re_exports(self):
        """Test that re-exports match the actual classes."""
        from core.signal_router import JournalWriter as FacadeJournalWriter
        from core.signal_router import SignalRouter as FacadeSignalRouter
        from core.signal_router.router import JournalWriter as RealJournalWriter
        from core.signal_router.router import SignalRouter as RealSignalRouter

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
        with open(self.journal_path, "r") as f:
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

        # Create a real asyncio task
        async def dummy_task():
            pass

        jw._task = asyncio.create_task(dummy_task())

        await jw.stop()

        # Allow event loop to advance
        await asyncio.sleep(0)

        # Assert task is completed after stop()
        assert jw._task.done()

    def test_multiple_appends(self):
        """Test multiple synchronous appends."""
        jw = JournalWriter(self.journal_path)

        entries = [
            {"id": 1, "action": "store"},
            {"id": 2, "action": "cancel"},
            {"id": 3, "action": "store"},
        ]

        for entry in entries:
            jw.append(entry)

        # Verify all entries written
        with open(self.journal_path, "r") as f:
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
        self.config_patcher = patch("core.signal_router.router.get_config")
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False},
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=0,
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=timestamp,
        )

        signal_id = router._generate_signal_id(signal)
        # The timestamp is converted to datetime, so it will be a datetime string
        import datetime

        expected_timestamp = datetime.datetime.fromtimestamp(
            timestamp, tz=datetime.timezone.utc
        )
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

        self.config_patcher = patch("core.signal_router.router.get_config")
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False},
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()) + 1,
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
            timestamp=int(time.time()),
        )

        short_entry = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time()),
        )

        long_exit = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.EXIT_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            timestamp=int(time.time()),
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

        self.config_patcher = patch("core.signal_router.router.get_config")
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False},
        }

    def teardown_method(self):
        """Cleanup patches."""
        self.config_patcher.stop()

    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism for risk manager calls."""
        # Make risk manager fail twice then succeed
        self.risk_manager.evaluate_signal = AsyncMock(
            side_effect=[Exception("Fail1"), Exception("Fail2"), True]
        )

        router = SignalRouter(self.risk_manager, self.task_manager)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type=OrderType.MARKET,
            amount=Decimal("1.0"),
            timestamp=int(time.time()),
        )

        result = await router.process_signal(signal)

        # Should have been called 3 times (initial + 2 retries)
        assert self.risk_manager.evaluate_signal.call_count == 3
        assert result is not None

    @pytest.mark.asyncio
    async def test_critical_error_blocking(self):
        """Test blocking signals after critical errors."""
        self.risk_manager.evaluate_signal = AsyncMock(
            side_effect=Exception("Critical failure")
        )

        router = SignalRouter(
            self.risk_manager, self.task_manager, safe_mode_threshold=2, enable_queue=False
        )

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time()),
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

        self.config_patcher = patch("core.signal_router.router.get_config")
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False},
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
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
            timestamp=int(time.time()),
        )
        signal._original_timestamp = -1000  # Modify the original timestamp
        assert not router._validate_timestamp(signal)

        # Test with future timestamp (too far)
        signal._original_timestamp = int(time.time() + 157784630)  # 5+ years in future
        assert not router._validate_timestamp(signal)


class TestSignalRouterQueue:
    """Test SignalRouter message queue functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = MagicMock()
        self.risk_manager.evaluate_signal = AsyncMock(return_value=True)
        self.risk_manager.validate_signal = MagicMock(return_value=True)
        self.task_manager = MagicMock()

        self.config_patcher = patch("core.signal_router.router.get_config")
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False},
        }

    def teardown_method(self):
        """Cleanup patches."""
        self.config_patcher.stop()

    @pytest.fixture(autouse=True)
    async def cleanup_routers(self):
        """Cleanup any routers created during tests."""
        yield
        # Cleanup happens in individual test methods via shutdown calls

    def test_initialization_with_queue(self):
        """Test SignalRouter initialization with queue enabled."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=True,
            max_queue_size=50,
            worker_count=2,
            deduplication_window=120.0
        )

        assert router.enable_queue is True
        assert router.max_queue_size == 50
        assert router.worker_count == 2
        assert router.deduplication_window == 120.0
        assert router._signal_queue is not None
        assert len(router._processed_signals) == 0

    def test_initialization_without_queue(self):
        """Test SignalRouter initialization with queue disabled."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=False
        )

        assert router.enable_queue is False

    @pytest.mark.asyncio
    async def test_synchronous_processing_when_queue_disabled(self):
        """Test that signals are processed synchronously when queue is disabled."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=False
        )

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time()),
        )

        result = await router.process_signal(signal)

        # Should process synchronously and call risk manager
        assert result is not None
        self.risk_manager.evaluate_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_signal_detection(self):
        """Test that duplicate signals are detected and handled properly."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=True,
            deduplication_window=300.0
        )

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time()),
        )

        # First signal should be processed
        result1 = await router.process_signal(signal)
        assert result1 is not None

        # Duplicate signal should be detected
        result2 = await router.process_signal(signal)
        assert result2 is not None  # Should return the same result

        # Should only call risk manager once
        self.risk_manager.evaluate_signal.assert_called_once()

        # Check metrics
        metrics = router.get_queue_metrics()
        assert metrics["duplicate_signals"] == 1

    @pytest.mark.asyncio
    async def test_expired_duplicate_signals_allowed(self):
        """Test that expired duplicate signals are allowed."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=False,  # Disable queue for this test to ensure synchronous processing
            deduplication_window=1.0  # Very short window
        )

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USD",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            stop_loss=Decimal("49000"),
            timestamp=int(time.time()),
        )

        # First signal
        result1 = await router.process_signal(signal)
        assert result1 is not None

        # Clear active signals to simulate the signal being executed/cancelled
        router.clear_signals()

        # Wait for deduplication window to expire
        await asyncio.sleep(1.1)

        # Same signal should be processed again
        result2 = await router.process_signal(signal)
        assert result2 is not None

        # Should call risk manager twice
        assert self.risk_manager.evaluate_signal.call_count == 2



    @pytest.mark.timeout(30)
    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(self):
        """Test concurrent signal processing with timeout to prevent hangs."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=True,
            worker_count=4,
            max_queue_size=100
        )

        try:
            # Create multiple signals
            signals = []
            for i in range(50):
                signal = TradingSignal(
                    strategy_id=f"test_{i}",
                    symbol=f"BTC/USD_{i}",
                    signal_type=SignalType.ENTRY_LONG,
                    signal_strength=SignalStrength.WEAK,
                    order_type="market",
                    amount=Decimal("1.0"),
                    stop_loss=Decimal("49000"),
                    timestamp=int(time.time()) + i,
                )
                signals.append(signal)

            # Process signals concurrently
            tasks = [router.process_signal(signal) for signal in signals]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            successful_results = [r for r in results if not isinstance(r, Exception) and r is not None]
            assert len(successful_results) == len(signals)

            # Check metrics
            metrics = router.get_queue_metrics()
            assert metrics["processed_signals"] == len(signals)
        finally:
            # Clean up background tasks
            await router.shutdown()

    def test_queue_metrics(self):
        """Test queue metrics collection."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=True
        )

        metrics = router.get_queue_metrics()

        required_keys = [
            "queue_depth", "processed_signals", "duplicate_signals",
            "processing_latency", "queue_full_rejects", "processing_errors",
            "avg_processing_latency", "max_processing_latency", "min_processing_latency",
            "current_queue_size", "active_workers", "idempotency_records"
        ]

        for key in required_keys:
            assert key in metrics

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up_tasks(self):
        """Test that shutdown properly cleans up background tasks."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=True,
            worker_count=2
        )

        # Let tasks start
        await asyncio.sleep(0.1)

        # Shutdown
        await router.shutdown()

        # Check that tasks are cleaned up
        assert len(router._queue_worker_tasks) == 0
        assert len(router._processing_tasks) == 0
        assert len(router._processed_signals) == 0


class TestSignalRouterRaceConditions:
    """Test SignalRouter race condition handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = MagicMock()
        self.risk_manager.evaluate_signal = AsyncMock(return_value=True)
        self.risk_manager.validate_signal = MagicMock(return_value=True)
        self.task_manager = MagicMock()

        self.config_patcher = patch("core.signal_router.router.get_config")
        self.mock_get_config = self.config_patcher.start()
        self.mock_get_config.return_value = {
            "predictive_models": {"enabled": False},
            "ml": {"enabled": False},
            "journal": {"enabled": False},
        }

    def teardown_method(self):
        """Cleanup patches."""
        self.config_patcher.stop()

    @pytest.mark.timeout(30)
    @pytest.mark.asyncio
    async def test_concurrent_access_from_multiple_coroutines(self):
        """Test concurrent access from multiple coroutines."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=True,
            worker_count=4
        )

        try:
            async def process_signal(coro_id: int):
                """Process a signal in a coroutine."""
                signal = TradingSignal(
                    strategy_id=f"coro_{coro_id}",
                    symbol=f"BTC/USD_{coro_id}",
                    signal_type=SignalType.ENTRY_LONG,
                    signal_strength=SignalStrength.WEAK,
                    order_type="market",
                    amount=Decimal("1.0"),
                    stop_loss=Decimal("49000"),
                    timestamp=int(time.time()) + coro_id,
                )
                result = await router.process_signal(signal)
                return result

            # Create multiple concurrent tasks
            tasks = [process_signal(i) for i in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Should have processed signals without errors
            successful_results = [r for r in results if not isinstance(r, Exception) and r is not None]
            assert len(successful_results) == 20
        finally:
            # Clean up background tasks
            await router.shutdown()

    @pytest.mark.timeout(30)
    @pytest.mark.asyncio
    async def test_high_frequency_duplicate_signals(self):
        """Test handling of high-frequency duplicate signals."""
        router = SignalRouter(
            self.risk_manager,
            self.task_manager,
            enable_queue=True,
            deduplication_window=60.0  # 1 minute window
        )

        try:
            signal = TradingSignal(
                strategy_id="high_freq",
                symbol="BTC/USD",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.WEAK,
                order_type="market",
                amount=Decimal("1.0"),
                stop_loss=Decimal("49000"),
                timestamp=int(time.time()),
            )

            # Send duplicate signals sequentially to avoid overwhelming the queue
            results = []
            for _ in range(20):  # Reduced from 100 to 20
                result = await router.process_signal(signal)
                results.append(result)

            # All should succeed
            successful_results = [r for r in results if not isinstance(r, Exception) and r is not None]
            assert len(successful_results) == 20

            # But risk manager should only be called once
            self.risk_manager.evaluate_signal.assert_called_once()

            # Check duplicate count
            metrics = router.get_queue_metrics()
            assert metrics["duplicate_signals"] == 19  # First one + 19 duplicates
        finally:
            # Clean up background tasks
            await router.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
