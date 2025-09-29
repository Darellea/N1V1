"""
Event definitions for the event-driven architecture.

This module defines the canonical set of events with structured payloads
that modules use to communicate via the event bus.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """Canonical event types for the system."""

    # Market and regime events
    REGIME_CHANGE = "regime_change"
    MARKET_DATA_UPDATE = "market_data_update"

    # Strategy events
    STRATEGY_SWITCH = "strategy_switch"
    STRATEGY_PERFORMANCE_UPDATE = "strategy_performance_update"

    # Trading events
    TRADE_EXECUTED = "trade_executed"
    TRADE_CANCELLED = "trade_cancelled"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"

    # Risk events
    RISK_LIMIT_TRIGGERED = "risk_limit_triggered"
    POSITION_SIZE_ADJUSTED = "position_size_adjusted"

    # System events
    DIAGNOSTIC_ALERT = "diagnostic_alert"
    SYSTEM_STATUS_UPDATE = "system_status_update"

    # Knowledge events
    KNOWLEDGE_ENTRY_CREATED = "knowledge_entry_created"
    KNOWLEDGE_QUERY_EXECUTED = "knowledge_query_executed"


@dataclass
class BaseEvent:
    """Base event class with common fields."""

    event_type: EventType
    source: str  # Component that generated the event
    timestamp: datetime
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata or {},
        }


@dataclass
class RegimeChangeEvent(BaseEvent):
    """Event fired when market regime changes."""

    def __init__(
        self,
        old_regime: str,
        new_regime: str,
        confidence: float,
        market_data: Optional[Dict[str, Any]] = None,
        source: str = "regime_detector",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.REGIME_CHANGE,
            source=source,
            timestamp=datetime.now(),
            payload={
                "old_regime": old_regime,
                "new_regime": new_regime,
                "confidence": confidence,
                "market_data": market_data,
            },
            metadata=metadata,
        )


@dataclass
class StrategySwitchEvent(BaseEvent):
    """Event fired when strategy selection changes."""

    def __init__(
        self,
        previous_strategy: Optional[str],
        new_strategy: str,
        rationale: str,
        confidence: float,
        market_conditions: Optional[Dict[str, Any]] = None,
        source: str = "strategy_selector",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.STRATEGY_SWITCH,
            source=source,
            timestamp=datetime.now(),
            payload={
                "previous_strategy": previous_strategy,
                "new_strategy": new_strategy,
                "rationale": rationale,
                "confidence": confidence,
                "market_conditions": market_conditions,
            },
            metadata=metadata,
        )


@dataclass
class TradeExecutedEvent(BaseEvent):
    """Event fired when a trade is executed."""

    def __init__(
        self,
        trade_id: str,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: Decimal,
        price: Decimal,
        slippage: Optional[Decimal] = None,
        commission: Optional[Decimal] = None,
        strategy: Optional[str] = None,
        source: str = "executor",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.TRADE_EXECUTED,
            source=source,
            timestamp=datetime.now(),
            payload={
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "quantity": str(quantity),
                "price": str(price),
                "slippage": str(slippage) if slippage else None,
                "commission": str(commission) if commission else None,
                "strategy": strategy,
            },
            metadata=metadata,
        )


@dataclass
class RiskLimitTriggeredEvent(BaseEvent):
    """Event fired when a risk limit is triggered."""

    def __init__(
        self,
        risk_factor: str,  # e.g., 'daily_loss', 'position_size', 'drawdown'
        trigger_condition: str,
        current_value: Any,
        threshold_value: Any,
        defensive_action: str,
        symbol: Optional[str] = None,
        source: str = "risk_manager",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.RISK_LIMIT_TRIGGERED,
            source=source,
            timestamp=datetime.now(),
            payload={
                "risk_factor": risk_factor,
                "trigger_condition": trigger_condition,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "defensive_action": defensive_action,
                "symbol": symbol,
            },
            metadata=metadata,
        )


@dataclass
class DiagnosticAlertEvent(BaseEvent):
    """Event fired for diagnostic alerts."""

    def __init__(
        self,
        alert_type: str,  # 'error', 'warning', 'info'
        component: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "diagnostics",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.DIAGNOSTIC_ALERT,
            source=source,
            timestamp=datetime.now(),
            payload={
                "alert_type": alert_type,
                "component": component,
                "message": message,
                "details": details,
            },
            metadata=metadata,
        )


@dataclass
class KnowledgeEntryCreatedEvent(BaseEvent):
    """Event fired when a new knowledge entry is created."""

    def __init__(
        self,
        entry_id: str,
        regime: str,
        strategy: str,
        performance_metrics: Dict[str, Any],
        outcome: str,
        source: str = "knowledge_base",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.KNOWLEDGE_ENTRY_CREATED,
            source=source,
            timestamp=datetime.now(),
            payload={
                "entry_id": entry_id,
                "regime": regime,
                "strategy": strategy,
                "performance_metrics": performance_metrics,
                "outcome": outcome,
            },
            metadata=metadata,
        )


@dataclass
class MarketDataUpdateEvent(BaseEvent):
    """Event fired when market data is updated."""

    def __init__(
        self,
        symbol: str,
        data_type: str,  # 'ohlcv', 'indicators', 'orderbook'
        data: Dict[str, Any],
        source: str = "market_data",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.MARKET_DATA_UPDATE,
            source=source,
            timestamp=datetime.now(),
            payload={"symbol": symbol, "data_type": data_type, "data": data},
            metadata=metadata,
        )


@dataclass
class SystemStatusUpdateEvent(BaseEvent):
    """Event fired for system status updates."""

    def __init__(
        self,
        component: str,
        status: str,  # 'starting', 'running', 'stopping', 'error'
        details: Optional[Dict[str, Any]] = None,
        source: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            event_type=EventType.SYSTEM_STATUS_UPDATE,
            source=source,
            timestamp=datetime.now(),
            payload={"component": component, "status": status, "details": details},
            metadata=metadata,
        )


# Convenience functions for creating events
def create_regime_change_event(
    old_regime: str,
    new_regime: str,
    confidence: float,
    market_data: Optional[Dict[str, Any]] = None,
) -> RegimeChangeEvent:
    """Create a regime change event."""
    return RegimeChangeEvent(
        old_regime=old_regime,
        new_regime=new_regime,
        confidence=confidence,
        market_data=market_data,
    )


def create_strategy_switch_event(
    previous_strategy: Optional[str],
    new_strategy: str,
    rationale: str,
    confidence: float,
    market_conditions: Optional[Dict[str, Any]] = None,
) -> StrategySwitchEvent:
    """Create a strategy switch event."""
    return StrategySwitchEvent(
        previous_strategy=previous_strategy,
        new_strategy=new_strategy,
        rationale=rationale,
        confidence=confidence,
        market_conditions=market_conditions,
    )


def create_trade_executed_event(
    trade_id: str,
    symbol: str,
    side: str,
    quantity: Decimal,
    price: Decimal,
    slippage: Optional[Decimal] = None,
    commission: Optional[Decimal] = None,
    strategy: Optional[str] = None,
) -> TradeExecutedEvent:
    """Create a trade executed event."""
    return TradeExecutedEvent(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        slippage=slippage,
        commission=commission,
        strategy=strategy,
    )


def create_risk_limit_triggered_event(
    risk_factor: str,
    trigger_condition: str,
    current_value: Any,
    threshold_value: Any,
    defensive_action: str,
    symbol: Optional[str] = None,
) -> RiskLimitTriggeredEvent:
    """Create a risk limit triggered event."""
    return RiskLimitTriggeredEvent(
        risk_factor=risk_factor,
        trigger_condition=trigger_condition,
        current_value=current_value,
        threshold_value=threshold_value,
        defensive_action=defensive_action,
        symbol=symbol,
    )


def create_diagnostic_alert_event(
    alert_type: str,
    component: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> DiagnosticAlertEvent:
    """Create a diagnostic alert event."""
    return DiagnosticAlertEvent(
        alert_type=alert_type, component=component, message=message, details=details
    )


def create_knowledge_entry_created_event(
    entry_id: str,
    regime: str,
    strategy: str,
    performance_metrics: Dict[str, Any],
    outcome: str,
) -> KnowledgeEntryCreatedEvent:
    """Create a knowledge entry created event."""
    return KnowledgeEntryCreatedEvent(
        entry_id=entry_id,
        regime=regime,
        strategy=strategy,
        performance_metrics=performance_metrics,
        outcome=outcome,
    )
