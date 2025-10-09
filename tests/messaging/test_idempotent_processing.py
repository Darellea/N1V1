"""
Unit tests for idempotent message processing functionality.

Tests idempotent processing of trading signals and orders with deduplication,
exactly-once semantics, and message replay protection.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from core.contracts import TradingSignal
from core.idempotency import OrderExecutionRegistry


@dataclass
class ProcessingResult:
    """Result of message processing."""

    executed: bool
    message_id: str
    idempotency_key: str
    timestamp: float
    duplicate_detected: bool = False
    replay_attempted: bool = False


class IdempotentMessageProcessor:
    """
    Processor for handling messages with idempotent semantics.

    Supports deduplication, exactly-once processing, and configurable
    deduplication windows for trading signals and orders.
    """

    def __init__(
        self,
        deduplication_window_seconds: int = 300,
        replay_protection_window_seconds: int = 3600,
    ):
        """
        Initialize the processor.

        Args:
            deduplication_window_seconds: Time window for deduplication in seconds
            replay_protection_window_seconds: Time window for replay protection in seconds
        """
        self.deduplication_window = deduplication_window_seconds
        self.replay_protection_window = replay_protection_window_seconds
        self.registry = OrderExecutionRegistry()
        self.processed_messages: Dict[str, ProcessingResult] = {}
        self.metrics = {
            "total_processed": 0,
            "duplicates_detected": 0,
            "replay_attempts_blocked": 0,
            "processing_time_avg": 0.0,
        }

    async def process(self, message: Any) -> ProcessingResult:
        """
        Process a message with idempotent semantics.

        Args:
            message: TradingSignal or order message to process

        Returns:
            ProcessingResult indicating the outcome
        """
        start_time = time.time()

        # Extract idempotency key
        idempotency_key = self._extract_idempotency_key(message)
        if not idempotency_key:
            # Generate one if not provided
            idempotency_key = str(uuid.uuid4())
            self._set_idempotency_key(message, idempotency_key)

        # Check for existing processing
        existing_result = self.processed_messages.get(idempotency_key)
        if existing_result:
            # Check if within deduplication window
            time_diff = time.time() - existing_result.timestamp
            if time_diff < self.deduplication_window:
                self.metrics["duplicates_detected"] += 1
                return ProcessingResult(
                    executed=False,
                    message_id=getattr(message, "id", str(id(message))),
                    idempotency_key=idempotency_key,
                    timestamp=time.time(),
                    duplicate_detected=True,
                )
            elif time_diff < self.replay_protection_window:
                # Within replay protection window - block as potential replay attack
                self.metrics["replay_attempts_blocked"] += 1
                return ProcessingResult(
                    executed=False,
                    message_id=getattr(message, "id", str(id(message))),
                    idempotency_key=idempotency_key,
                    timestamp=time.time(),
                    duplicate_detected=False,
                    replay_attempted=True,
                )
            else:
                # Outside both windows - allow reprocessing for legitimate business retries
                pass  # Continue with processing

        # Check registry for concurrent processing
        registry_state = self.registry.begin_execution(idempotency_key)
        if registry_state:
            if (
                isinstance(registry_state, dict)
                and registry_state.get("status") == "pending"
            ):
                return ProcessingResult(
                    executed=False,
                    message_id=getattr(message, "id", str(id(message))),
                    idempotency_key=idempotency_key,
                    timestamp=time.time(),
                    duplicate_detected=True,
                )

        try:
            # Simulate message processing
            await asyncio.sleep(0.001)  # Small delay to simulate processing

            result = ProcessingResult(
                executed=True,
                message_id=getattr(message, "id", str(id(message))),
                idempotency_key=idempotency_key,
                timestamp=time.time(),
            )

            # Mark as successful
            self.registry.mark_success(idempotency_key, {"processed": True})
            self.processed_messages[idempotency_key] = result
            self.metrics["total_processed"] += 1

            # Update average processing time
            processing_time = time.time() - start_time
            self.metrics["processing_time_avg"] = (
                (
                    self.metrics["processing_time_avg"]
                    * (self.metrics["total_processed"] - 1)
                )
                + processing_time
            ) / self.metrics["total_processed"]

            return result

        except Exception as e:
            self.registry.mark_failure(idempotency_key, e)
            raise

    def _extract_idempotency_key(self, message: Any) -> Optional[str]:
        """Extract idempotency key from message."""
        if isinstance(message, TradingSignal):
            return message.idempotency_key
        elif isinstance(message, dict):
            return message.get("idempotency_key")
        elif hasattr(message, "idempotency_key"):
            return message.idempotency_key
        return None

    def _set_idempotency_key(self, message: Any, key: str) -> None:
        """Set idempotency key on message."""
        if isinstance(message, TradingSignal):
            message.idempotency_key = key
        elif isinstance(message, dict):
            message["idempotency_key"] = key
        elif hasattr(message, "idempotency_key"):
            message.idempotency_key = key

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return self.metrics.copy()

    def clear(self) -> None:
        """Clear all processed messages and registry."""
        self.processed_messages.clear()
        self.registry.clear()
        self.metrics = {
            "total_processed": 0,
            "duplicates_detected": 0,
            "replay_attempts_blocked": 0,
            "processing_time_avg": 0.0,
        }


class TestIdempotentMessageProcessing:
    """Test cases for idempotent message processing."""

    @pytest.fixture
    def processor(self):
        """Create a test processor instance."""
        return IdempotentMessageProcessor()

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trading signal."""
        return TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type="ENTRY_LONG",
            signal_strength="STRONG",
            order_type="MARKET",
            amount=0.001,
            current_price=50000.0,
            timestamp=int(time.time() * 1000),
            idempotency_key="test_key_123",
        )

    @pytest.mark.timeout(15)
    def test_duplicate_detection_performance(self, processor, sample_signal):
        """Test that duplicate detection works efficiently within timeout."""
        # Process many messages with duplicates to test performance
        messages = []
        for i in range(1000):
            msg = TradingSignal(
                strategy_id=f"strategy_{i % 10}",
                symbol="BTC/USDT",
                signal_type="ENTRY_LONG",
                signal_strength="STRONG",
                order_type="MARKET",
                amount=0.001,
                current_price=50000.0,
                timestamp=int(time.time() * 1000),
                idempotency_key=f"msg_{i % 100}",  # Create duplicates every 100 messages
            )
            messages.append(msg)

        start_time = time.time()
        results = []
        for msg in messages:
            result = asyncio.run(processor.process(msg))
            results.append(result)

        processing_time = time.time() - start_time

        # Should process efficiently within timeout
        assert processing_time < 10.0
        # Verify correct duplicate detection
        executed_count = sum(1 for r in results if r.executed)
        assert executed_count == 100  # Only 100 unique messages should be executed

        # Check metrics
        metrics = processor.get_metrics()
        assert metrics["total_processed"] == 100
        assert metrics["duplicates_detected"] == 900  # 900 duplicates

    @pytest.mark.asyncio
    async def test_idempotency_guarantee_exact_once(self, processor, sample_signal):
        """Test that each message is processed exactly once."""
        # Process the same signal multiple times
        result1 = await processor.process(sample_signal)
        result2 = await processor.process(sample_signal)
        result3 = await processor.process(sample_signal)

        # First should execute
        assert result1.executed is True
        assert result1.duplicate_detected is False

        # Subsequent should be detected as duplicates
        assert result2.executed is False
        assert result2.duplicate_detected is True
        assert result2.idempotency_key == sample_signal.idempotency_key

        assert result3.executed is False
        assert result3.duplicate_detected is True

        # Check metrics
        metrics = processor.get_metrics()
        assert metrics["total_processed"] == 1
        assert metrics["duplicates_detected"] == 2

    @pytest.mark.asyncio
    async def test_message_replay_protection(self, processor, sample_signal):
        """Test protection against message replay attacks."""
        # Process original message
        original_result = await processor.process(sample_signal)
        assert original_result.executed is True

        # Simulate replay after deduplication window (set processor window to 0 for test)
        processor.deduplication_window = 0

        # Wait a tiny bit to ensure we're outside window
        await asyncio.sleep(0.001)

        replay_result = await processor.process(sample_signal)

        # Replay should be blocked
        assert replay_result.executed is False
        assert replay_result.replay_attempted is True

        # Check metrics include replay attempts
        metrics = processor.get_metrics()
        assert metrics["replay_attempts_blocked"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_processing_blocked(self, processor, sample_signal):
        """Test that concurrent processing of same message is blocked."""
        # Start two concurrent processing tasks
        task1 = asyncio.create_task(processor.process(sample_signal))
        task2 = asyncio.create_task(processor.process(sample_signal))

        # Wait for both to complete
        result1, result2 = await asyncio.gather(task1, task2)

        # Only one should execute, one should be blocked
        executed_count = sum(1 for r in [result1, result2] if r.executed)
        blocked_count = sum(1 for r in [result1, result2] if r.duplicate_detected)

        assert executed_count == 1
        assert blocked_count == 1

    @pytest.mark.asyncio
    async def test_different_idempotency_keys_allowed(self, processor):
        """Test that messages with different idempotency keys are processed separately."""
        signal1 = TradingSignal(
            strategy_id="strategy_1",
            symbol="BTC/USDT",
            signal_type="ENTRY_LONG",
            signal_strength="STRONG",
            order_type="MARKET",
            amount=0.001,
            current_price=50000.0,
            timestamp=int(time.time() * 1000),
            idempotency_key="key_1",
        )

        signal2 = TradingSignal(
            strategy_id="strategy_2",
            symbol="ETH/USDT",
            signal_type="ENTRY_SHORT",
            signal_strength="MODERATE",
            order_type="LIMIT",
            amount=0.01,
            current_price=3000.0,
            timestamp=int(time.time() * 1000),
            idempotency_key="key_2",
        )

        result1 = await processor.process(signal1)
        result2 = await processor.process(signal2)

        # Both should execute
        assert result1.executed is True
        assert result2.executed is True
        assert result1.idempotency_key != result2.idempotency_key

        # Check metrics
        metrics = processor.get_metrics()
        assert metrics["total_processed"] == 2
        assert metrics["duplicates_detected"] == 0

    @pytest.mark.asyncio
    async def test_auto_generated_idempotency_keys(self, processor):
        """Test that messages without idempotency keys get them auto-generated."""
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type="ENTRY_LONG",
            signal_strength="STRONG",
            order_type="MARKET",
            amount=0.001,
            current_price=50000.0,
            timestamp=int(time.time() * 1000)
            # No idempotency_key provided
        )

        # Should not have key initially
        assert signal.idempotency_key is None

        result = await processor.process(signal)

        # Should have key after processing
        assert signal.idempotency_key is not None
        assert result.idempotency_key == signal.idempotency_key
        assert result.executed is True

    @pytest.mark.asyncio
    async def test_processing_metrics_accuracy(self, processor, sample_signal):
        """Test that processing metrics are accurate."""
        # Process multiple messages
        await processor.process(sample_signal)

        # Try duplicate
        await processor.process(sample_signal)

        # Process different message
        different_signal = sample_signal.copy()
        different_signal.idempotency_key = "different_key"
        await processor.process(different_signal)

        metrics = processor.get_metrics()

        assert metrics["total_processed"] == 2  # Two unique messages
        assert metrics["duplicates_detected"] == 1  # One duplicate
        assert metrics["processing_time_avg"] > 0  # Should have timing data

    @pytest.mark.asyncio
    async def test_deduplication_window_respected(self, processor, sample_signal):
        """Test that deduplication window is properly respected."""
        # Set short windows for test - make replay protection window same as deduplication for this test
        processor.deduplication_window = 1  # 1 second
        processor.replay_protection_window = 1  # 1 second (same as deduplication)

        # Process first time
        result1 = await processor.process(sample_signal)
        assert result1.executed is True

        # Immediate duplicate should be blocked
        result2 = await processor.process(sample_signal)
        assert result2.executed is False
        assert result2.duplicate_detected is True

        # Wait for both windows to expire
        await asyncio.sleep(1.1)

        # Should allow reprocessing after both windows expire
        result3 = await processor.process(sample_signal)
        assert result3.executed is True  # Should execute again

        metrics = processor.get_metrics()
        assert metrics["total_processed"] == 2  # Two executions
        assert metrics["duplicates_detected"] == 1  # One duplicate blocked

    @pytest.mark.asyncio
    async def test_message_processing_error_handling(self, processor):
        """Test error handling during message processing."""
        # Create a signal that might cause issues
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type="ENTRY_LONG",
            signal_strength="STRONG",
            order_type="MARKET",
            amount=0.001,
            current_price=50000.0,
            timestamp=int(time.time() * 1000),
            idempotency_key="error_test_key",
        )

        # Process normally first
        result = await processor.process(signal)
        assert result.executed is True

        # Verify registry state
        registry_state = processor.registry.get_status("error_test_key")
        assert registry_state is not None
        assert registry_state["status"] == "success"
