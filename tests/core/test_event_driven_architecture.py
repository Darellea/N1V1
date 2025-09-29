"""
Test suite for the event-driven architecture.

Tests cover:
- Event definitions and creation
- Event bus functionality (publish/subscribe)
- Enhanced event bus features
- Event-driven logging integration
- Error handling and edge cases
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from core.signal_router.event_bus import (
    EnhancedEventBus,
    EventBus,
    create_enhanced_event_bus,
    get_default_enhanced_event_bus,
    get_default_event_bus,
)
from core.signal_router.events import (
    BaseEvent,
    EventType,
    create_diagnostic_alert_event,
    create_knowledge_entry_created_event,
    create_regime_change_event,
    create_risk_limit_triggered_event,
    create_strategy_switch_event,
    create_trade_executed_event,
)
from utils.logger import TradeLogger


class TestEventDefinitions:
    """Test event definitions and creation."""

    def test_base_event_creation(self):
        """Test creating a base event."""
        event = BaseEvent(
            event_type=EventType.REGIME_CHANGE,
            source="test_component",
            timestamp=datetime.now(),
            payload={"test": "data"},
            metadata={"extra": "info"},
        )

        assert event.event_type == EventType.REGIME_CHANGE
        assert event.source == "test_component"
        assert isinstance(event.timestamp, datetime)
        assert event.payload == {"test": "data"}
        assert event.metadata == {"extra": "info"}

    def test_regime_change_event(self):
        """Test regime change event creation."""
        event = create_regime_change_event(
            old_regime="trending",
            new_regime="sideways",
            confidence=0.85,
            market_data={"volatility": 0.15},
        )

        assert event.event_type == EventType.REGIME_CHANGE
        assert event.payload["old_regime"] == "trending"
        assert event.payload["new_regime"] == "sideways"
        assert event.payload["confidence"] == 0.85
        assert event.payload["market_data"] == {"volatility": 0.15}

    def test_strategy_switch_event(self):
        """Test strategy switch event creation."""
        event = create_strategy_switch_event(
            previous_strategy="RSI",
            new_strategy="MACD",
            rationale="Better performance in current regime",
            confidence=0.92,
        )

        assert event.event_type == EventType.STRATEGY_SWITCH
        assert event.payload["previous_strategy"] == "RSI"
        assert event.payload["new_strategy"] == "MACD"
        assert event.payload["rationale"] == "Better performance in current regime"
        assert event.payload["confidence"] == 0.92

    def test_trade_executed_event(self):
        """Test trade executed event creation."""
        event = create_trade_executed_event(
            trade_id="12345",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            slippage=Decimal("5"),
            commission=Decimal("2.5"),
            strategy="RSIStrategy",
        )

        assert event.event_type == EventType.TRADE_EXECUTED
        assert event.payload["trade_id"] == "12345"
        assert event.payload["symbol"] == "BTC/USDT"
        assert event.payload["side"] == "buy"
        assert event.payload["quantity"] == "0.1"
        assert event.payload["price"] == "50000"
        assert event.payload["slippage"] == "5"
        assert event.payload["commission"] == "2.5"
        assert event.payload["strategy"] == "RSIStrategy"

    def test_risk_limit_triggered_event(self):
        """Test risk limit triggered event creation."""
        event = create_risk_limit_triggered_event(
            risk_factor="daily_loss",
            trigger_condition="exceeded_threshold",
            current_value=150.0,
            threshold_value=100.0,
            defensive_action="reduce_position_size",
            symbol="BTC/USDT",
        )

        assert event.event_type == EventType.RISK_LIMIT_TRIGGERED
        assert event.payload["risk_factor"] == "daily_loss"
        assert event.payload["trigger_condition"] == "exceeded_threshold"
        assert event.payload["current_value"] == 150.0
        assert event.payload["threshold_value"] == 100.0
        assert event.payload["defensive_action"] == "reduce_position_size"
        assert event.payload["symbol"] == "BTC/USDT"

    def test_diagnostic_alert_event(self):
        """Test diagnostic alert event creation."""
        event = create_diagnostic_alert_event(
            alert_type="warning",
            component="strategy_selector",
            message="High volatility detected",
            details={"volatility": 0.25, "threshold": 0.20},
        )

        assert event.event_type == EventType.DIAGNOSTIC_ALERT
        assert event.payload["alert_type"] == "warning"
        assert event.payload["component"] == "strategy_selector"
        assert event.payload["message"] == "High volatility detected"
        assert event.payload["details"] == {"volatility": 0.25, "threshold": 0.20}

    def test_knowledge_entry_created_event(self):
        """Test knowledge entry created event creation."""
        event = create_knowledge_entry_created_event(
            entry_id="kb_001",
            regime="trending",
            strategy="RSIStrategy",
            performance_metrics={"win_rate": 0.65, "sharpe": 1.2},
            outcome="success",
        )

        assert event.event_type == EventType.KNOWLEDGE_ENTRY_CREATED
        assert event.payload["entry_id"] == "kb_001"
        assert event.payload["regime"] == "trending"
        assert event.payload["strategy"] == "RSIStrategy"
        assert event.payload["performance_metrics"] == {"win_rate": 0.65, "sharpe": 1.2}
        assert event.payload["outcome"] == "success"

    def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = create_regime_change_event("trending", "sideways", 0.8)
        event_dict = event.to_dict()

        assert event_dict["event_type"] == "regime_change"
        assert event_dict["source"] == "regime_detector"
        assert "timestamp" in event_dict
        assert event_dict["payload"]["old_regime"] == "trending"
        assert event_dict["payload"]["new_regime"] == "sideways"
        assert event_dict["payload"]["confidence"] == 0.8


class TestEventBus:
    """Test basic event bus functionality."""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for each test."""
        return EventBus()

    @pytest.fixture
    def sample_event(self):
        """Create a sample event for testing."""
        from core.signal_router.event_bus import SignalEvent

        return SignalEvent(
            signal=MagicMock(),
            event_type="test_event",
            source="test_source",
            metadata={"test": "metadata"},
        )

    def test_event_bus_initialization(self, event_bus):
        """Test event bus initialization."""
        assert event_bus._subscribers == {}
        assert event_bus._event_history == []
        assert event_bus._max_history == 1000

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, event_bus, sample_event):
        """Test basic subscribe and publish functionality."""
        received_events = []

        async def test_handler(event):
            received_events.append(event)

        # Subscribe to the event
        await event_bus.subscribe("test_event", test_handler)

        # Publish the event
        await event_bus.publish(sample_event)

        # Verify the event was received
        assert len(received_events) == 1
        assert received_events[0] == sample_event

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus, sample_event):
        """Test multiple subscribers to the same event."""
        received_count = {"count": 0}

        async def handler1(event):
            received_count["count"] += 1

        async def handler2(event):
            received_count["count"] += 1

        # Subscribe both handlers
        await event_bus.subscribe("test_event", handler1)
        await event_bus.subscribe("test_event", handler2)

        # Publish the event
        await event_bus.publish(sample_event)

        # Verify both handlers were called
        assert received_count["count"] == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus, sample_event):
        """Test unsubscribing from events."""
        received_events = []

        async def test_handler(event):
            received_events.append(event)

        # Subscribe and then unsubscribe
        await event_bus.subscribe("test_event", test_handler)
        await event_bus.unsubscribe("test_event", test_handler)

        # Publish the event
        await event_bus.publish(sample_event)

        # Verify no events were received
        assert len(received_events) == 0

    def test_get_event_history(self, event_bus, sample_event):
        """Test retrieving event history."""
        # Add event to history manually for testing
        event_bus._event_history.append(sample_event)

        # Test getting all history
        history = event_bus.get_event_history()
        assert len(history) == 1
        assert history[0] == sample_event

        # Test filtering by event type
        history_filtered = event_bus.get_event_history("test_event")
        assert len(history_filtered) == 1

        # Test filtering by non-existent event type
        history_empty = event_bus.get_event_history("non_existent")
        assert len(history_empty) == 0

    def test_get_subscriber_count(self, event_bus):
        """Test getting subscriber counts."""
        # Initially no subscribers
        assert event_bus.get_subscriber_count() == 0
        assert event_bus.get_subscriber_count("test_event") == 0

        # Add a subscriber
        event_bus._subscribers["test_event"] = [MagicMock()]
        assert event_bus.get_subscriber_count() == 1
        assert event_bus.get_subscriber_count("test_event") == 1


class TestEnhancedEventBus:
    """Test enhanced event bus functionality."""

    @pytest.fixture
    def enhanced_event_bus(self):
        """Create a fresh enhanced event bus for each test."""
        return EnhancedEventBus()

    @pytest.fixture
    def sample_base_event(self):
        """Create a sample base event for testing."""
        return create_regime_change_event("trending", "sideways", 0.8)

    def test_enhanced_event_bus_initialization(self, enhanced_event_bus):
        """Test enhanced event bus initialization."""
        assert enhanced_event_bus.async_mode is True
        assert enhanced_event_bus.log_all_events is True
        assert enhanced_event_bus.buffer_size == 1000
        assert enhanced_event_bus._event_count == 0
        assert enhanced_event_bus._dropped_events == 0
        assert enhanced_event_bus._error_count == 0

    @pytest.mark.asyncio
    async def test_publish_base_event(self, enhanced_event_bus, sample_base_event):
        """Test publishing a BaseEvent."""
        received_events = []

        async def test_handler(event):
            received_events.append(event)

        # Subscribe to the event
        await enhanced_event_bus.subscribe(EventType.REGIME_CHANGE, test_handler)

        # Publish the event
        await enhanced_event_bus.publish(sample_base_event)

        # Verify the event was received
        assert len(received_events) == 1
        assert received_events[0] == sample_base_event
        assert enhanced_event_bus._event_count == 1

    @pytest.mark.asyncio
    async def test_publish_with_string_event_type(
        self, enhanced_event_bus, sample_base_event
    ):
        """Test subscribing with string event type."""
        received_events = []

        async def test_handler(event):
            received_events.append(event)

        # Subscribe using string
        await enhanced_event_bus.subscribe("regime_change", test_handler)

        # Publish the event
        await enhanced_event_bus.publish(sample_base_event)

        # Verify the event was received
        assert len(received_events) == 1

    @pytest.mark.asyncio
    async def test_event_history_management(self, enhanced_event_bus):
        """Test event history buffer management."""
        # Create many events
        events = []
        for i in range(5):
            event = create_regime_change_event(f"regime_{i}", f"regime_{i+1}", 0.8)
            events.append(event)
            await enhanced_event_bus.publish(event)

        # Check history
        history = enhanced_event_bus.get_event_history()
        assert len(history) == 5

        # Test filtering
        history_filtered = enhanced_event_bus.get_event_history(EventType.REGIME_CHANGE)
        assert len(history_filtered) == 5

    @pytest.mark.asyncio
    async def test_error_handling_in_publish(
        self, enhanced_event_bus, sample_base_event
    ):
        """Test error handling during event publishing."""

        async def failing_handler(event):
            raise Exception("Test error")

        # Subscribe the failing handler
        await enhanced_event_bus.subscribe(EventType.REGIME_CHANGE, failing_handler)

        # Publish should not raise, but should increment error count
        await enhanced_event_bus.publish(sample_base_event)

        # Check that error was recorded
        assert enhanced_event_bus._error_count == 1

    def test_get_stats(self, enhanced_event_bus):
        """Test getting event bus statistics."""
        stats = enhanced_event_bus.get_stats()

        expected_keys = [
            "total_events",
            "events_in_history",
            "dropped_events",
            "error_count",
            "subscribers",
            "event_counts",
            "buffer_size",
            "async_mode",
        ]

        for key in expected_keys:
            assert key in stats

        assert stats["total_events"] == 0
        assert stats["async_mode"] is True

    @pytest.mark.asyncio
    async def test_clear_subscribers(self, enhanced_event_bus):
        """Test clearing all subscribers."""

        async def test_handler(event):
            pass

        # Add subscribers
        await enhanced_event_bus.subscribe(EventType.REGIME_CHANGE, test_handler)
        await enhanced_event_bus.subscribe(EventType.STRATEGY_SWITCH, test_handler)

        assert enhanced_event_bus.get_subscriber_count() == 2

        # Clear subscribers
        await enhanced_event_bus.clear_subscribers()

        assert enhanced_event_bus.get_subscriber_count() == 0


class TestEventDrivenLogging:
    """Test event-driven logging integration."""

    @pytest.fixture
    def trade_logger(self):
        """Create a trade logger for testing."""
        return TradeLogger("test_logger")

    @pytest.mark.asyncio
    async def test_trade_logger_event_handling(self, trade_logger):
        """Test that trade logger can handle events."""
        # Create a trade executed event
        event = create_trade_executed_event(
            trade_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )

        # Handle the event
        await trade_logger.handle_event(event)

        # Check that trade was recorded
        trades = trade_logger.get_trade_history()
        assert len(trades) == 1
        assert trades[0]["pair"] == "BTC/USDT"
        assert trades[0]["action"] == "buy"

    @pytest.mark.asyncio
    async def test_strategy_switch_event_handling(self, trade_logger):
        """Test strategy switch event handling."""
        event = create_strategy_switch_event(
            previous_strategy="RSI",
            new_strategy="MACD",
            rationale="Better performance",
            confidence=0.9,
        )

        # Handle the event (should not raise)
        await trade_logger.handle_event(event)

        # Verify the logger handled it without errors
        assert True  # If we get here, the event was handled

    @pytest.mark.asyncio
    async def test_risk_limit_event_handling(self, trade_logger):
        """Test risk limit triggered event handling."""
        event = create_risk_limit_triggered_event(
            risk_factor="daily_loss",
            trigger_condition="exceeded",
            current_value=150.0,
            threshold_value=100.0,
            defensive_action="reduce_positions",
        )

        # Handle the event (should not raise)
        await trade_logger.handle_event(event)

        # Verify the logger handled it without errors
        assert True

    @pytest.mark.asyncio
    async def test_diagnostic_alert_event_handling(self, trade_logger):
        """Test diagnostic alert event handling."""
        event = create_diagnostic_alert_event(
            alert_type="warning",
            component="strategy_selector",
            message="High volatility detected",
        )

        # Handle the event (should not raise)
        await trade_logger.handle_event(event)

        # Verify the logger handled it without errors
        assert True

    @pytest.mark.asyncio
    async def test_knowledge_entry_event_handling(self, trade_logger):
        """Test knowledge entry created event handling."""
        event = create_knowledge_entry_created_event(
            entry_id="kb_001",
            regime="trending",
            strategy="RSIStrategy",
            performance_metrics={"win_rate": 0.7},
            outcome="success",
        )

        # Handle the event (should not raise)
        await trade_logger.handle_event(event)

        # Verify the logger handled it without errors
        assert True

    @pytest.mark.asyncio
    async def test_regime_change_event_handling(self, trade_logger):
        """Test regime change event handling."""
        event = create_regime_change_event(
            old_regime="trending", new_regime="sideways", confidence=0.85
        )

        # Handle the event (should not raise)
        await trade_logger.handle_event(event)

        # Verify the logger handled it without errors
        assert True

    @pytest.mark.asyncio
    async def test_system_status_event_handling(self, trade_logger):
        """Test system status update event handling."""
        event = BaseEvent(
            event_type=EventType.SYSTEM_STATUS_UPDATE,
            source="system_monitor",
            timestamp=datetime.now(),
            payload={
                "component": "strategy_selector",
                "status": "running",
                "details": {"uptime": "1h 30m"},
            },
        )

        # Handle the event (should not raise)
        await trade_logger.handle_event(event)

        # Verify the logger handled it without errors
        assert True

    @pytest.mark.asyncio
    async def test_unknown_event_handling(self, trade_logger):
        """Test handling of unknown event types."""
        # Create an event with unknown type by using a mock
        event = BaseEvent(
            event_type=MagicMock(),  # Mock to simulate unknown type
            source="test",
            timestamp=datetime.now(),
            payload={},
        )

        # Mock the event_type.value to return unknown
        event.event_type.value = "unknown_event_type"

        # Handle the event (should not raise)
        await trade_logger.handle_event(event)

        # Verify the logger handled it without errors
        assert True


class TestGlobalInstances:
    """Test global event bus instances."""

    def test_get_default_event_bus(self):
        """Test getting the default event bus."""
        bus = get_default_event_bus()
        assert isinstance(bus, EventBus)

    def test_get_default_enhanced_event_bus(self):
        """Test getting the default enhanced event bus."""
        bus = get_default_enhanced_event_bus()
        assert isinstance(bus, EnhancedEventBus)

    def test_create_enhanced_event_bus(self):
        """Test creating a custom enhanced event bus."""
        config = {"async_mode": False, "buffer_size": 500, "log_all_events": False}
        bus = create_enhanced_event_bus(config)

        assert isinstance(bus, EnhancedEventBus)
        assert bus.async_mode is False
        assert bus.buffer_size == 500
        assert bus.log_all_events is False


class TestIntegration:
    """Test integration between components."""

    @pytest.mark.asyncio
    async def test_end_to_end_event_flow(self):
        """Test complete event flow from creation to handling."""
        # Create event bus
        event_bus = EnhancedEventBus()

        # Create logger
        logger = TradeLogger("integration_test")

        # Subscribe logger to events
        await event_bus.subscribe(EventType.TRADE_EXECUTED, logger.handle_event)

        # Create and publish event
        event = create_trade_executed_event(
            trade_id="integration_001",
            symbol="ETH/USDT",
            side="sell",
            quantity=Decimal("1.0"),
            price=Decimal("3000"),
        )

        await event_bus.publish(event)

        # Verify event was processed
        trades = logger.get_trade_history()
        assert len(trades) == 1
        assert trades[0]["pair"] == "ETH/USDT"
        assert trades[0]["action"] == "sell"

        # Check event bus stats
        stats = event_bus.get_stats()
        assert stats["total_events"] == 1
        assert stats["subscribers"] == 1

    @pytest.mark.asyncio
    async def test_multiple_event_types(self):
        """Test handling multiple event types simultaneously."""
        event_bus = EnhancedEventBus()
        logger = TradeLogger("multi_test")

        # Subscribe to multiple event types
        event_types = [
            EventType.TRADE_EXECUTED,
            EventType.STRATEGY_SWITCH,
            EventType.RISK_LIMIT_TRIGGERED,
        ]

        for event_type in event_types:
            await event_bus.subscribe(event_type, logger.handle_event)

        # Publish different events
        events = [
            create_trade_executed_event(
                "t1", "BTC/USDT", "buy", Decimal("0.1"), Decimal("50000")
            ),
            create_strategy_switch_event("RSI", "MACD", "Better fit", 0.9),
            create_risk_limit_triggered_event(
                "daily_loss", "exceeded", 120.0, 100.0, "pause_trading"
            ),
        ]

        for event in events:
            await event_bus.publish(event)

        # Verify all events were processed
        stats = event_bus.get_stats()
        assert stats["total_events"] == 3

        # Check that trade was recorded
        trades = logger.get_trade_history()
        assert len(trades) == 1


if __name__ == "__main__":
    pytest.main([__file__])
