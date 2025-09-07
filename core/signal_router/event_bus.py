"""
Event bus for event-driven architecture using publish/subscribe pattern.

Provides a centralized mechanism for routing events between all system components
including strategies, risk managers, executors, knowledge base, and monitoring.
"""

import asyncio
import logging
import json
from typing import Dict, List, Callable, Any, Optional, Awaitable, Union
from dataclasses import dataclass
from datetime import datetime
from core.contracts import TradingSignal
from .events import BaseEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class SignalEvent:
    """
    Represents a signal routing event.
    """
    signal: TradingSignal
    event_type: str  # 'new_signal', 'signal_approved', 'signal_rejected', etc.
    source: str  # Component that generated the event
    target: Optional[str] = None  # Target component (None for broadcast)
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = asyncio.get_event_loop().time()


class EventBus:
    """
    Publish/subscribe event bus for signal routing.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[SignalEvent] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()

    async def publish(self, event: SignalEvent) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
        """
        async with self._lock:
            # Store in history
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

            # Notify subscribers
            event_type = event.event_type
            if event_type in self._subscribers:
                tasks = []
                for subscriber in self._subscribers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            tasks.append(subscriber(event))
                        else:
                            # Run sync subscriber in thread pool
                            loop = asyncio.get_event_loop()
                            tasks.append(loop.run_in_executor(None, subscriber, event))
                    except Exception as e:
                        logger.exception(f"Error notifying subscriber: {e}")

                # Wait for all notifications to complete
                if tasks:
                    try:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    except Exception as e:
                        logger.exception(f"Error in event notification: {e}")

            logger.debug(f"Published event: {event.event_type} from {event.source}")

    async def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of event to subscribe to
            callback: Callback function to handle the event
        """
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type} events")

    async def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        async with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from {event_type} events")
                except ValueError:
                    logger.warning(f"Callback not found in subscribers for {event_type}")

    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[SignalEvent]:
        """
        Get recent event history.

        Args:
            event_type: Optional event type filter
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    async def clear_subscribers(self) -> None:
        """
        Clear all subscribers (useful for testing or shutdown).
        """
        async with self._lock:
            self._subscribers.clear()
            logger.info("Cleared all event subscribers")

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """
        Get the number of subscribers.

        Args:
            event_type: Optional event type filter

        Returns:
            Number of subscribers
        """
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(subs) for subs in self._subscribers.values())


class EnhancedEventBus:
    """
    Enhanced event bus supporting the full event-driven architecture.

    Features:
    - Support for structured events (BaseEvent and subclasses)
    - Async and sync operation modes
    - Configurable buffer size and error handling
    - Event serialization and logging
    - Thread-safe operations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced event bus.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.async_mode = self.config.get('async_mode', True)
        self.log_all_events = self.config.get('log_all_events', True)
        self.buffer_size = self.config.get('buffer_size', 1000)

        # Subscriber registry
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Union[SignalEvent, BaseEvent]] = []
        self._lock = asyncio.Lock()

        # Performance tracking
        self._event_count = 0
        self._dropped_events = 0
        self._error_count = 0

        logger.info("EnhancedEventBus initialized")

    async def publish(self, event: Union[SignalEvent, BaseEvent]) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish (SignalEvent or BaseEvent)
        """
        async with self._lock:
            try:
                # Store in history with buffer management
                self._event_history.append(event)
                if len(self._event_history) > self.buffer_size:
                    self._event_history.pop(0)
                    self._dropped_events += 1

                self._event_count += 1

                # Get event type for subscription lookup
                if isinstance(event, BaseEvent):
                    event_type = event.event_type.value
                else:
                    event_type = event.event_type

                # Notify subscribers
                if event_type in self._subscribers:
                    tasks = []
                    for subscriber in self._subscribers[event_type]:
                        try:
                            if self.async_mode and asyncio.iscoroutinefunction(subscriber):
                                tasks.append(subscriber(event))
                            elif not self.async_mode:
                                # Run sync subscriber in thread pool
                                loop = asyncio.get_event_loop()
                                tasks.append(loop.run_in_executor(None, subscriber, event))
                            else:
                                # Async mode but sync subscriber - run in thread pool
                                loop = asyncio.get_event_loop()
                                tasks.append(loop.run_in_executor(None, subscriber, event))
                        except Exception as e:
                            logger.exception(f"Error queuing subscriber notification: {e}")
                            self._error_count += 1

                    # Execute notifications
                    if tasks:
                        try:
                            if self.async_mode:
                                # For async mode, still wait for completion to ensure test reliability
                                # In production, this could be fire-and-forget
                                await self._notify_subscribers_async(tasks)
                            else:
                                # Wait for completion in sync mode
                                results = await asyncio.gather(*tasks, return_exceptions=True)
                                # Check for exceptions in results
                                for result in results:
                                    if isinstance(result, Exception):
                                        logger.exception(f"Error in subscriber notification: {result}")
                                        self._error_count += 1
                        except Exception as e:
                            logger.exception(f"Error in event notification: {e}")
                            self._error_count += 1

                # Log event if configured
                if self.log_all_events:
                    await self._log_event(event)

                logger.debug(f"Published event: {event_type}")

            except Exception as e:
                logger.exception(f"Error publishing event: {e}")
                self._error_count += 1

    async def _notify_subscribers_async(self, tasks: List[Awaitable]) -> None:
        """
        Notify subscribers asynchronously (fire and forget).

        Args:
            tasks: List of notification tasks
        """
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Check for exceptions in results
            for result in results:
                if isinstance(result, Exception):
                    logger.exception(f"Error in async subscriber notification: {result}")
                    self._error_count += 1
        except Exception as e:
            logger.exception(f"Error in async subscriber notification: {e}")
            self._error_count += 1

    async def subscribe(self, event_type: Union[str, EventType], callback: Callable) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of event to subscribe to (string or EventType)
            callback: Callback function to handle the event
        """
        async with self._lock:
            # Convert EventType to string if needed
            if isinstance(event_type, EventType):
                event_type = event_type.value

            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type} events")

    async def unsubscribe(self, event_type: Union[str, EventType], callback: Callable) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        async with self._lock:
            # Convert EventType to string if needed
            if isinstance(event_type, EventType):
                event_type = event_type.value

            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed from {event_type} events")
                except ValueError:
                    logger.warning(f"Callback not found in subscribers for {event_type}")

    async def publish_event(self, event: BaseEvent) -> None:
        """
        Convenience method to publish a BaseEvent.

        Args:
            event: The BaseEvent to publish
        """
        await self.publish(event)

    def get_event_history(
        self,
        event_type: Optional[Union[str, EventType]] = None,
        limit: int = 100
    ) -> List[Union[SignalEvent, BaseEvent]]:
        """
        Get recent event history.

        Args:
            event_type: Optional event type filter
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        events = self._event_history

        if event_type:
            # Convert EventType to string if needed
            if isinstance(event_type, EventType):
                event_type = event_type.value

            if isinstance(event_type, str):
                events = [e for e in events if
                         (isinstance(e, BaseEvent) and e.event_type.value == event_type) or
                         (isinstance(e, SignalEvent) and e.event_type == event_type)]

        return events[-limit:]

    async def clear_subscribers(self) -> None:
        """
        Clear all subscribers (useful for testing or shutdown).
        """
        async with self._lock:
            self._subscribers.clear()
            logger.info("Cleared all event subscribers")

    def get_subscriber_count(self, event_type: Optional[Union[str, EventType]] = None) -> int:
        """
        Get the number of subscribers.

        Args:
            event_type: Optional event type filter

        Returns:
            Number of subscribers
        """
        if event_type:
            # Convert EventType to string if needed
            if isinstance(event_type, EventType):
                event_type = event_type.value
            return len(self._subscribers.get(event_type, []))
        return sum(len(subs) for subs in self._subscribers.values())

    async def _log_event(self, event: Union[SignalEvent, BaseEvent]) -> None:
        """
        Log an event for monitoring and debugging.

        Args:
            event: The event to log
        """
        try:
            if isinstance(event, BaseEvent):
                log_data = {
                    "event_type": event.event_type.value,
                    "source": event.source,
                    "timestamp": event.timestamp.isoformat(),
                    "payload": event.payload
                }
                if event.metadata:
                    log_data["metadata"] = event.metadata

                logger.info(f"Event: {event.event_type.value}", extra={"event_data": log_data})
            else:
                # Legacy SignalEvent logging
                logger.info(f"Signal Event: {event.event_type} from {event.source}")

        except Exception as e:
            logger.exception(f"Error logging event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Dictionary with statistics
        """
        event_counts = {}
        for event in self._event_history:
            if isinstance(event, BaseEvent):
                event_type = event.event_type.value
            else:
                event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        return {
            "total_events": self._event_count,
            "events_in_history": len(self._event_history),
            "dropped_events": self._dropped_events,
            "error_count": self._error_count,
            "subscribers": self.get_subscriber_count(),
            "event_counts": event_counts,
            "buffer_size": self.buffer_size,
            "async_mode": self.async_mode
        }




class SignalRouter:
    """
    High-level signal router using the event bus.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the signal router.

        Args:
            event_bus: Optional event bus instance
        """
        self.event_bus = event_bus or EventBus()
        self._components: Dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the signal router and set up event subscriptions.
        """
        if self._initialized:
            return

        # Subscribe to signal events
        await self.event_bus.subscribe("new_signal", self._handle_new_signal)
        await self.event_bus.subscribe("signal_approved", self._handle_signal_approved)
        await self.event_bus.subscribe("signal_rejected", self._handle_signal_rejected)

        self._initialized = True
        logger.info("SignalRouter initialized")

    async def register_component(self, name: str, component: Any) -> None:
        """
        Register a component with the router.

        Args:
            name: Component name
            component: Component instance
        """
        self._components[name] = component
        logger.info(f"Registered component: {name}")

    async def route_signal(self, signal: TradingSignal, source: str = "unknown") -> None:
        """
        Route a signal through the event bus.

        Args:
            signal: The trading signal to route
            source: Source component name
        """
        if not self._initialized:
            await self.initialize()

        event = SignalEvent(
            signal=signal,
            event_type="new_signal",
            source=source,
            metadata={"routing_start": asyncio.get_event_loop().time()}
        )

        await self.event_bus.publish(event)

    async def _handle_new_signal(self, event: SignalEvent) -> None:
        """
        Handle new signal events.

        Args:
            event: The signal event
        """
        try:
            # Route to risk manager first
            if "risk_manager" in self._components:
                await self._route_to_risk_manager(event)

            # Route to other components based on signal characteristics
            await self._route_to_components(event)

        except Exception as e:
            logger.exception(f"Error handling new signal: {e}")
            # Publish rejection event
            reject_event = SignalEvent(
                signal=event.signal,
                event_type="signal_rejected",
                source="signal_router",
                metadata={"reason": "processing_error", "error": str(e)}
            )
            await self.event_bus.publish(reject_event)

    async def _route_to_risk_manager(self, event: SignalEvent) -> None:
        """
        Route signal to risk manager for validation.

        Args:
            event: The signal event
        """
        risk_manager = self._components.get("risk_manager")
        if not risk_manager:
            return

        try:
            # Call risk manager evaluation
            approved = await risk_manager.evaluate_signal(event.signal)

            if approved:
                # Publish approval event
                approve_event = SignalEvent(
                    signal=event.signal,
                    event_type="signal_approved",
                    source="risk_manager",
                    metadata={"approved_by": "risk_manager"}
                )
                await self.event_bus.publish(approve_event)
            else:
                # Publish rejection event
                reject_event = SignalEvent(
                    signal=event.signal,
                    event_type="signal_rejected",
                    source="risk_manager",
                    metadata={"reason": "risk_check_failed"}
                )
                await self.event_bus.publish(reject_event)

        except Exception as e:
            logger.exception(f"Risk manager evaluation failed: {e}")
            # Publish rejection event
            reject_event = SignalEvent(
                signal=event.signal,
                event_type="signal_rejected",
                source="risk_manager",
                metadata={"reason": "risk_error", "error": str(e)}
            )
            await self.event_bus.publish(reject_event)

    async def _route_to_components(self, event: SignalEvent) -> None:
        """
        Route signal to other registered components.

        Args:
            event: The signal event
        """
        for name, component in self._components.items():
            if name == "risk_manager":
                continue  # Already handled

            try:
                # Check if component has a signal handler
                if hasattr(component, 'handle_signal'):
                    await component.handle_signal(event.signal)
                elif hasattr(component, 'process_signal'):
                    await component.process_signal(event.signal)
                elif callable(component):
                    # Assume it's a callable that accepts signals
                    if asyncio.iscoroutinefunction(component):
                        await component(event.signal)
                    else:
                        # Run in thread pool
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, component, event.signal)

            except Exception as e:
                logger.exception(f"Error routing to {name}: {e}")

    async def _handle_signal_approved(self, event: SignalEvent) -> None:
        """
        Handle signal approval events.

        Args:
            event: The approval event
        """
        logger.info(f"Signal approved: {event.signal.symbol} from {event.source}")

        # Route to execution components
        for name, component in self._components.items():
            if name in ["executor", "order_manager"]:
                try:
                    if hasattr(component, 'execute_signal'):
                        await component.execute_signal(event.signal)
                    elif hasattr(component, 'handle_signal'):
                        await component.handle_signal(event.signal)
                except Exception as e:
                    logger.exception(f"Error executing signal on {name}: {e}")

    async def _handle_signal_rejected(self, event: SignalEvent) -> None:
        """
        Handle signal rejection events.

        Args:
            event: The rejection event
        """
        reason = event.metadata.get("reason", "unknown") if event.metadata else "unknown"
        logger.info(f"Signal rejected: {event.signal.symbol} from {event.source}, reason: {reason}")

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics.

        Returns:
            Dictionary with routing statistics
        """
        history = self.event_bus.get_event_history()

        stats = {
            "total_events": len(history),
            "subscribers": self.event_bus.get_subscriber_count(),
            "components": list(self._components.keys())
        }

        # Count events by type
        event_counts = {}
        for event in history:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        stats["event_counts"] = event_counts
        return stats


# Global event bus instances
default_event_bus = EventBus()
default_enhanced_event_bus = EnhancedEventBus()

# Convenience functions
async def publish_signal_event(signal: TradingSignal, event_type: str, source: str,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Convenience function to publish a signal event.

    Args:
        signal: The trading signal
        event_type: Type of event
        source: Source component
        metadata: Optional metadata
    """
    event = SignalEvent(
        signal=signal,
        event_type=event_type,
        source=source,
        metadata=metadata
    )
    await default_event_bus.publish(event)


async def publish_event(event: BaseEvent) -> None:
    """
    Convenience function to publish a structured event.

    Args:
        event: The BaseEvent to publish
    """
    await default_enhanced_event_bus.publish_event(event)


def get_default_event_bus() -> EventBus:
    """
    Get the default event bus instance.

    Returns:
        Default EventBus instance
    """
    return default_event_bus


def get_default_enhanced_event_bus() -> EnhancedEventBus:
    """
    Get the default enhanced event bus instance.

    Returns:
        Default EnhancedEventBus instance
    """
    return default_enhanced_event_bus


def create_signal_router(event_bus: Optional[EventBus] = None) -> SignalRouter:
    """
    Create a new signal router instance.

    Args:
        event_bus: Optional event bus instance

    Returns:
        SignalRouter instance
    """
    return SignalRouter(event_bus)


def create_enhanced_event_bus(config: Optional[Dict[str, Any]] = None) -> EnhancedEventBus:
    """
    Create a new enhanced event bus instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        EnhancedEventBus instance
    """
    return EnhancedEventBus(config)
