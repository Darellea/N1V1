"""
Event bus for signal routing using publish/subscribe pattern.

Provides a centralized mechanism for routing signals between strategies,
risk managers, executors, and other components.
"""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional, Awaitable
from dataclasses import dataclass
from core.contracts import TradingSignal

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


# Global event bus instance
default_event_bus = EventBus()

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


def get_default_event_bus() -> EventBus:
    """
    Get the default event bus instance.

    Returns:
        Default EventBus instance
    """
    return default_event_bus


def create_signal_router(event_bus: Optional[EventBus] = None) -> SignalRouter:
    """
    Create a new signal router instance.

    Args:
        event_bus: Optional event bus instance

    Returns:
        SignalRouter instance
    """
    return SignalRouter(event_bus)
