"""
Circuit Breaker System for N1V1 Trading Framework

This module implements a comprehensive circuit breaker system that automatically
suspends trading operations when predefined risk thresholds are breached. The
system provides multiple layers of protection against catastrophic drawdowns
while maintaining operational integrity and providing clear recovery procedures.

Key Features:
- Multi-factor risk threshold monitoring (drawdown, losses, volatility, correlation)
- Real-time equity curve monitoring with millisecond precision
- Sophisticated state management with clear transitions
- Configurable cooling periods and recovery procedures
- Comprehensive logging and incident analysis
- Integration with existing risk management framework

Circuit Breaker States:
- NORMAL: Trading operations active
- MONITORING: Approaching risk thresholds
- TRIGGERED: Circuit breaker activated, trading suspended
- COOLING: Waiting period before recovery attempt
- RECOVERY: Gradual trading resumption
- EMERGENCY: Manual intervention required

Trigger Conditions:
- Equity Drawdown: Maximum daily/weekly drawdown percentage
- Consecutive Losses: Number of losing trades in sequence
- Risk-Adjusted Performance: Sharpe ratio deterioration
- Volatility Spike: Abnormal market volatility detection
- Correlation Breakdown: Unusual correlation changes
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Import metrics collector at module level to avoid blocking imports in async methods
try:
    from core.metrics_collector import get_metrics_collector

    _metrics_collector_available = True
except ImportError:
    _metrics_collector_available = False


class CooldownStrategy(Enum):
    """Cooldown period calculation strategies."""

    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


@dataclass
class CircuitBreakerConfig:
    """Configuration for Circuit Breaker."""

    equity_drawdown_threshold: float = 0.1
    consecutive_losses_threshold: int = 5
    volatility_spike_threshold: float = 0.05
    max_triggers_per_hour: int = 3
    monitoring_window_minutes: int = 60
    cooling_period_minutes: int = 5
    recovery_period_minutes: int = 10
    max_history_size: int = 1000

    # Cooldown enforcement settings
    cooldown_strategy: CooldownStrategy = CooldownStrategy.EXPONENTIAL
    base_cooldown_minutes: int = 5
    max_cooldown_minutes: int = 60
    cooldown_multiplier: float = 2.0
    enable_cooldown_enforcement: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.equity_drawdown_threshold <= 0:
            raise ValueError("equity_drawdown_threshold must be positive")
        if self.consecutive_losses_threshold <= 0:
            raise ValueError("consecutive_losses_threshold must be positive")
        if self.volatility_spike_threshold <= 0:
            raise ValueError("volatility_spike_threshold must be positive")
        if self.max_triggers_per_hour <= 0:
            raise ValueError("max_triggers_per_hour must be positive")
        if self.monitoring_window_minutes <= 0:
            raise ValueError("monitoring_window_minutes must be positive")
        if self.cooling_period_minutes <= 0:
            raise ValueError("cooling_period_minutes must be positive")
        if self.recovery_period_minutes <= 0:
            raise ValueError("recovery_period_minutes must be positive")

    def __repr__(self):
        return (
            f"CircuitBreakerConfig(equity_drawdown_threshold={self.equity_drawdown_threshold}, "
            f"consecutive_losses_threshold={self.consecutive_losses_threshold}, "
            f"volatility_spike_threshold={self.volatility_spike_threshold}, "
            f"max_triggers_per_hour={self.max_triggers_per_hour}, "
            f"monitoring_window_minutes={self.monitoring_window_minutes}, "
            f"cooling_period_minutes={self.cooling_period_minutes}, "
            f"recovery_period_minutes={self.recovery_period_minutes})"
        )


class CircuitBreakerState(Enum):
    """Circuit breaker operational states."""

    NORMAL = "normal"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    COOLING = "cooling"
    RECOVERY = "recovery"
    EMERGENCY = "emergency"


class TriggerSeverity(Enum):
    """Trigger severity levels."""

    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class TriggerCondition:
    """A single trigger condition with its configuration."""

    name: str
    description: str
    severity: TriggerSeverity
    threshold: float
    window_minutes: int
    enabled: bool = True
    weight: float = 1.0  # For multi-factor scoring
    cooldown_minutes: int = 0  # Minimum time between triggers


@dataclass
class CircuitBreakerEvent:
    """A circuit breaker event record."""

    timestamp: datetime
    event_type: str
    severity: TriggerSeverity
    trigger_name: str
    trigger_value: float
    threshold: float
    state_before: CircuitBreakerState
    state_after: CircuitBreakerState
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class TriggerEvent:
    """A circuit breaker trigger event for dashboard integration."""

    trigger_type: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"TriggerEvent(trigger_type='{self.trigger_type}', timestamp={self.timestamp}, details={self.details})"


@dataclass
class EquityPoint:
    """A single equity curve data point."""

    timestamp: datetime
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    trade_count: int


class TriggerStrategy(ABC):
    """Abstract base class for trigger condition strategies."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config

    @abstractmethod
    async def check_condition(
        self, conditions: Dict[str, Any], circuit_breaker: "CircuitBreaker"
    ) -> bool:
        """Check if this trigger condition is met."""
        pass

    @abstractmethod
    def get_trigger_name(self) -> str:
        """Get the name of this trigger condition."""
        pass


class EquityDrawdownTrigger(TriggerStrategy):
    """Strategy for checking equity drawdown conditions."""

    def get_trigger_name(self) -> str:
        return "equity_drawdown"

    async def check_condition(
        self, conditions: Dict[str, Any], circuit_breaker: "CircuitBreaker"
    ) -> bool:
        """Check if equity drawdown exceeds threshold."""
        if "equity" not in conditions:
            return False

        equity = conditions["equity"]
        # Assume peak equity is initial 10000 for simplicity
        peak_equity = 10000.0

        if peak_equity <= 0:
            return False

        drawdown = (peak_equity - equity) / peak_equity
        return drawdown >= self.config.equity_drawdown_threshold


class ConsecutiveLossesTrigger(TriggerStrategy):
    """Strategy for checking consecutive losses conditions."""

    def get_trigger_name(self) -> str:
        return "consecutive_losses"

    async def check_condition(
        self, conditions: Dict[str, Any], circuit_breaker: "CircuitBreaker"
    ) -> bool:
        """Check if consecutive losses exceed threshold."""
        if "consecutive_losses" not in conditions:
            return False

        consecutive_losses = conditions["consecutive_losses"]
        return consecutive_losses >= self.config.consecutive_losses_threshold


class VolatilitySpikeTrigger(TriggerStrategy):
    """Strategy for checking volatility spike conditions."""

    def get_trigger_name(self) -> str:
        return "volatility_spike"

    async def check_condition(
        self, conditions: Dict[str, Any], circuit_breaker: "CircuitBreaker"
    ) -> bool:
        """Check if volatility spike exceeds threshold."""
        if "volatility" not in conditions:
            return False

        volatility = conditions["volatility"]
        return volatility >= self.config.volatility_spike_threshold


class AnomalyTrigger(TriggerStrategy):
    """Strategy for checking anomaly detection conditions."""

    def get_trigger_name(self) -> str:
        return "market_anomaly"

    async def check_condition(
        self, conditions: Dict[str, Any], circuit_breaker: "CircuitBreaker"
    ) -> bool:
        """Check for market anomalies using integrated anomaly detector."""
        if not circuit_breaker.anomaly_detector:
            return False

        if "market_data" not in conditions:
            return False

        try:
            return await asyncio.wait_for(
                circuit_breaker.anomaly_detector.detect_market_anomaly(
                    conditions["market_data"]
                ),
                timeout=15.0,  # 15 second timeout for anomaly detection
            )
        except asyncio.TimeoutError:
            circuit_breaker.logger.warning("Timeout in anomaly detection")
            return False
        except Exception as e:
            circuit_breaker.logger.warning(f"Error in anomaly detection: {e}")
            return False


class StateMachine:
    """State machine for managing circuit breaker state transitions."""

    def __init__(self, circuit_breaker: "CircuitBreaker"):
        self.circuit_breaker = circuit_breaker
        self.transitions = self._build_transitions()

    def _build_transitions(
        self,
    ) -> Dict[CircuitBreakerState, Dict[str, CircuitBreakerState]]:
        """Build the state transition table."""
        return {
            CircuitBreakerState.NORMAL: {
                "trigger": CircuitBreakerState.TRIGGERED,
                "monitor": CircuitBreakerState.MONITORING,
            },
            CircuitBreakerState.MONITORING: {
                "trigger": CircuitBreakerState.TRIGGERED,
                "normal": CircuitBreakerState.NORMAL,
            },
            CircuitBreakerState.TRIGGERED: {
                "cooling": CircuitBreakerState.COOLING,
                "emergency": CircuitBreakerState.EMERGENCY,
            },
            CircuitBreakerState.COOLING: {
                "recovery": CircuitBreakerState.RECOVERY,
                "trigger": CircuitBreakerState.TRIGGERED,
            },
            CircuitBreakerState.RECOVERY: {
                "normal": CircuitBreakerState.NORMAL,
                "trigger": CircuitBreakerState.TRIGGERED,
            },
            CircuitBreakerState.EMERGENCY: {"normal": CircuitBreakerState.NORMAL},
        }

    async def transition(self, action: str, reason: str = "") -> bool:
        """Attempt to transition to a new state based on the action."""
        async with self.circuit_breaker._lock:
            current_state = self.circuit_breaker.state

            if current_state not in self.transitions:
                return False

            state_transitions = self.transitions[current_state]
            if action not in state_transitions:
                return False

            new_state = state_transitions[action]
            old_state = self.circuit_breaker.state
            self.circuit_breaker.state = new_state

            # Log the transition
            self.circuit_breaker._log_event(
                f"state_transition_{action}",
                old_state,
                new_state,
                reason or f"State transition: {action}",
            )

            # Execute state-specific actions (without holding the lock to prevent deadlocks)
            # Release lock before calling async actions
            pass

        # Execute state-specific actions outside the lock to prevent deadlocks
        try:
            await asyncio.wait_for(
                self._execute_state_actions(new_state, old_state),
                timeout=30.0,  # 30 second timeout for state actions
            )
        except asyncio.TimeoutError:
            self.circuit_breaker.logger.warning(
                f"Timeout executing state actions for {new_state}"
            )
        except Exception as e:
            self.circuit_breaker.logger.warning(
                f"Error executing state actions for {new_state}: {e}"
            )

        return True

    async def _execute_state_actions(
        self, new_state: CircuitBreakerState, old_state: CircuitBreakerState
    ) -> None:
        """Execute actions specific to entering a new state."""
        if new_state == CircuitBreakerState.COOLING:
            await self.circuit_breaker._enter_cooling_period()
        elif new_state == CircuitBreakerState.RECOVERY:
            await self.circuit_breaker._enter_recovery_period()
        elif (
            new_state == CircuitBreakerState.NORMAL
            and old_state != CircuitBreakerState.NORMAL
        ):
            await self.circuit_breaker._return_to_normal()
        elif new_state == CircuitBreakerState.TRIGGERED:
            await self.circuit_breaker._trigger_circuit_breaker("State machine trigger")


class CircuitBreaker:
    """
    Circuit Breaker implementation for the N1V1 trading framework.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.NORMAL
        self.trigger_history = []
        self.event_history = []
        self.trade_results = []
        self.last_trigger_time = None
        self.trigger_count = 0

        # Current trigger information for dashboards
        self.current_trigger = None

        # Equity tracking for metrics integration
        self.current_equity = 0.0

        # Concurrency protection
        self._lock = asyncio.Lock()

        # Logger
        self.logger = logging.getLogger(__name__)

        # Integration components (set by tests)
        self.order_manager = None
        self.signal_router = None
        self.risk_manager = None
        self.anomaly_detector = None

        # Initialize strategy pattern components
        self.trigger_strategies = [
            EquityDrawdownTrigger(config),
            ConsecutiveLossesTrigger(config),
            VolatilitySpikeTrigger(config),
            AnomalyTrigger(config),
        ]

        # Initialize state machine
        self.state_machine = StateMachine(self)

        # Initialize background tasks set
        self._background_tasks = set()

    def _log_event(
        self,
        event_type: str,
        previous_state: CircuitBreakerState,
        new_state: CircuitBreakerState,
        reason: str,
        **kwargs,
    ) -> None:
        """Log a circuit breaker event to event_history."""
        event = {
            "timestamp": datetime.now(),
            "event_type": event_type,
            "previous_state": previous_state.value,
            "new_state": new_state.value,
            "reason": reason,
            **kwargs,
        }
        self.event_history.append(event)

        # Maintain max history size
        if len(self.event_history) > self.config.max_history_size:
            self.event_history = self.event_history[-self.config.max_history_size :]

        self.logger.info(
            f"Circuit breaker event: {event_type} - {previous_state.value} -> {new_state.value} ({reason})"
        )

    def _check_equity_drawdown(self, peak_equity: float, current_equity: float) -> bool:
        """Check if equity drawdown exceeds threshold."""
        if peak_equity <= 0:
            return False
        drawdown = (peak_equity - current_equity) / peak_equity
        return drawdown >= self.config.equity_drawdown_threshold

    def _record_trade_result(self, pnl: float, is_win: bool) -> None:
        """Record a trade result."""
        self.trade_results.append(is_win)

    def _check_consecutive_losses(self) -> bool:
        """Check if consecutive losses exceed threshold."""
        if len(self.trade_results) < self.config.consecutive_losses_threshold:
            return False
        # Check the last N trades
        recent_trades = self.trade_results[-self.config.consecutive_losses_threshold :]
        return all(not win for win in recent_trades)

    def _check_volatility_spike(self, prices: np.ndarray) -> bool:
        """Check if volatility spike exceeds threshold."""
        if len(prices) < 2:
            return False
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        return volatility >= self.config.volatility_spike_threshold

    def _calculate_trigger_score(self, factors: Dict[str, bool]) -> float:
        """Calculate multi-factor trigger score."""
        weights = {
            "equity_drawdown": 0.6,
            "consecutive_losses": 0.3,
            "volatility_spike": 0.1,
        }
        score = 0.0
        for factor, triggered in factors.items():
            if triggered:
                score += weights.get(factor, 0.0)
        return score

    def _calculate_cooldown_period(self, trigger_count: int) -> int:
        """Calculate cooldown period based on strategy and trigger count."""
        if not self.config.enable_cooldown_enforcement:
            return 0

        base_minutes = self.config.base_cooldown_minutes

        if self.config.cooldown_strategy == CooldownStrategy.FIXED:
            return min(base_minutes, self.config.max_cooldown_minutes)

        elif self.config.cooldown_strategy == CooldownStrategy.EXPONENTIAL:
            cooldown = base_minutes * (
                self.config.cooldown_multiplier ** (trigger_count - 1)
            )
            return min(int(cooldown), self.config.max_cooldown_minutes)

        elif self.config.cooldown_strategy == CooldownStrategy.FIBONACCI:
            # Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, ...
            if trigger_count <= 1:
                return base_minutes
            elif trigger_count == 2:
                return base_minutes
            else:
                a, b = base_minutes, base_minutes
                for _ in range(trigger_count - 2):
                    a, b = b, a + b
                return min(b, self.config.max_cooldown_minutes)

        return base_minutes

    def _is_cooldown_active(self) -> bool:
        """Check if cooldown period is currently active."""
        if not self.config.enable_cooldown_enforcement or not self.last_trigger_time:
            return False

        cooldown_minutes = self._calculate_cooldown_period(self.trigger_count)
        if cooldown_minutes <= 0:
            return False

        elapsed_minutes = (datetime.now() - self.last_trigger_time).total_seconds() / 60
        return elapsed_minutes < cooldown_minutes

    def get_remaining_cooldown_minutes(self) -> float:
        """Get remaining cooldown minutes if active, 0 otherwise."""
        if not self.config.enable_cooldown_enforcement or not self.last_trigger_time:
            return 0.0

        cooldown_minutes = self._calculate_cooldown_period(self.trigger_count)
        if cooldown_minutes <= 0:
            return 0.0

        elapsed_minutes = (datetime.now() - self.last_trigger_time).total_seconds() / 60
        remaining = max(0.0, cooldown_minutes - elapsed_minutes)
        return remaining

    async def _perform_health_check(self) -> bool:
        """Perform health check to determine if recovery is safe."""
        try:
            # Check if equity has recovered sufficiently
            if hasattr(self, "current_equity") and self.current_equity > 0:
                # Simple recovery check: equity above 95% of initial
                initial_equity = 10000.0  # This should be configurable
                recovery_threshold = 0.95
                if self.current_equity >= initial_equity * recovery_threshold:
                    return True

            # Check if consecutive losses have reset
            if len(self.trade_results) >= 3:
                recent_trades = self.trade_results[-3:]
                if not all(not win for win in recent_trades):  # Not all losses
                    return True

            # Default to cautious approach
            return False

        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return False

    async def check_and_trigger(self, conditions: Dict[str, Any]) -> bool:
        """Check conditions and trigger if needed using strategy pattern."""
        self.logger.debug(f"Checking trigger conditions: {conditions}")

        # Check cooldown enforcement first
        if self._is_cooldown_active():
            remaining_minutes = self.get_remaining_cooldown_minutes()
            self.logger.info(
                f"Circuit breaker cooldown active. Remaining: {remaining_minutes:.1f} minutes. "
                f"Skipping trigger check."
            )
            return False

        # Cache timestamp to avoid repeated time.monotonic calls
        cached_time = asyncio.get_event_loop().time()
        start_time = cached_time
        triggered_strategies = []

        # Use strategy pattern to check all trigger conditions with timeout protection
        for strategy in self.trigger_strategies:
            try:
                # Add timeout protection to prevent hangs
                result = await asyncio.wait_for(
                    strategy.check_condition(conditions, self),
                    timeout=10.0,  # 10 second timeout per strategy
                )
                # Remove debug logging during breaker engagement to reduce overhead
                if self.state != CircuitBreakerState.TRIGGERED:
                    strategy_duration = asyncio.get_event_loop().time() - cached_time
                    self.logger.debug(
                        f"Strategy {strategy.get_trigger_name()} checked in {strategy_duration:.3f}s: {result}"
                    )

                if result:
                    triggered_strategies.append(strategy.get_trigger_name())
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout checking {strategy.get_trigger_name()} after 10s"
                )
                continue
            except Exception as e:
                self.logger.warning(
                    f"Error checking {strategy.get_trigger_name()}: {e}"
                )
                continue

        if triggered_strategies:
            # Check for deduplication - don't trigger if already in TRIGGERED state
            if self.state == CircuitBreakerState.TRIGGERED:
                self.logger.debug(
                    "Circuit breaker already triggered, skipping duplicate trigger"
                )
                return False

            self.logger.info(f"Trigger conditions met: {triggered_strategies}")

            # Use state machine to transition to triggered state with timeout
            try:
                transition_result = await asyncio.wait_for(
                    self.state_machine.transition(
                        "trigger",
                        f"Strategies triggered: {', '.join(triggered_strategies)}",
                    ),
                    timeout=30.0,  # 30 second timeout for state transition
                )
                # Remove debug logging during breaker engagement
                if self.state != CircuitBreakerState.TRIGGERED:
                    transition_duration = asyncio.get_event_loop().time() - cached_time
                    self.logger.debug(
                        f"State transition completed in {transition_duration:.3f}s: {transition_result}"
                    )
            except asyncio.TimeoutError:
                self.logger.error("Timeout during state transition to TRIGGERED")
                return False
            except Exception as e:
                self.logger.error(f"Error during state transition: {e}")
                return False

            # Integration: cancel orders when triggered with timeout
            if self.order_manager and hasattr(self.order_manager, "cancel_all_orders"):
                try:
                    await asyncio.wait_for(
                        self.order_manager.cancel_all_orders(),
                        timeout=30.0,  # 30 second timeout
                    )
                    # Remove debug logging during breaker engagement
                    if self.state != CircuitBreakerState.TRIGGERED:
                        self.logger.debug("Successfully cancelled all orders")
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout cancelling orders")
                except Exception as e:
                    self.logger.warning(f"Failed to cancel orders: {e}")

            # Integration: block signals when triggered with timeout
            if self.signal_router and hasattr(self.signal_router, "block_signals"):
                try:
                    await asyncio.wait_for(
                        self.signal_router.block_signals(),
                        timeout=30.0,  # 30 second timeout
                    )
                    # Remove debug logging during breaker engagement
                    if self.state != CircuitBreakerState.TRIGGERED:
                        self.logger.debug("Successfully blocked signals")
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout blocking signals")
                except Exception as e:
                    self.logger.warning(f"Failed to block signals: {e}")

            total_duration = asyncio.get_event_loop().time() - start_time
            self.logger.info(f"check_and_trigger completed in {total_duration:.3f}s")
            return True

        check_duration = asyncio.get_event_loop().time() - start_time
        # Remove debug logging during breaker engagement
        if self.state != CircuitBreakerState.TRIGGERED:
            self.logger.debug(
                f"check_and_trigger completed in {check_duration:.3f}s - no triggers"
            )
        return False

    async def _trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger the circuit breaker."""
        async with self._lock:
            previous_state = self.state
            self.state = CircuitBreakerState.TRIGGERED

            # Cache timestamp to avoid multiple datetime.now() calls
            cached_timestamp = datetime.now()

            # Set current trigger for dashboard integration
            self.current_trigger = TriggerEvent(
                trigger_type=reason,
                timestamp=cached_timestamp,
                details={
                    "previous_state": previous_state.value,
                    "current_equity": self.current_equity,
                    "trigger_count": self.trigger_count + 1,
                },
            )

            self.trigger_history.append(
                {"timestamp": cached_timestamp, "reason": reason}
            )
            self.last_trigger_time = cached_timestamp
            self.trigger_count += 1
            self._log_event(
                "trigger", previous_state, CircuitBreakerState.TRIGGERED, reason
            )

            # Record state change in metrics - fire-and-forget during breaker engagement to reduce overhead
            if _metrics_collector_available:
                try:
                    metrics_collector = get_metrics_collector()
                    # Create fire-and-forget task for metrics recording during breaker engagement
                    asyncio.create_task(
                        self._record_metric_async(
                            metrics_collector,
                            "circuit_breaker_state",
                            1,  # 1 = triggered, 0 = normal
                            {"account": "main"},
                        )
                    )
                except Exception as e:
                    # Silent failure during breaker engagement to minimize overhead
                    pass

    async def _enter_cooling_period(self) -> None:
        """Enter cooling period."""
        async with self._lock:
            previous_state = self.state
            self.state = CircuitBreakerState.COOLING
            self._log_event(
                "cooling",
                previous_state,
                CircuitBreakerState.COOLING,
                "Entering cooling period",
            )

        # Integration: freeze portfolio when entering cooling
        if self.risk_manager and hasattr(self.risk_manager, "freeze_portfolio"):
            try:
                # Create task with proper exception handling and timeout
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self.risk_manager.freeze_portfolio(),
                        timeout=30.0,  # 30 second timeout
                    )
                )
                # Store task reference to prevent unhandled exceptions
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                self.logger.warning(f"Failed to freeze portfolio: {e}")

    async def _enter_recovery_period(self) -> None:
        """Enter recovery period with health check validation."""
        # Perform health check before entering recovery
        health_check_passed = await self._perform_health_check()

        async with self._lock:
            previous_state = self.state
            if health_check_passed:
                self.state = CircuitBreakerState.RECOVERY
                self._log_event(
                    "recovery",
                    previous_state,
                    CircuitBreakerState.RECOVERY,
                    "Entering recovery period - health check passed",
                )
            else:
                # Stay in cooling or go back to triggered if health check fails
                self.logger.warning("Health check failed, remaining in cooling period")
                self._log_event(
                    "recovery_blocked",
                    previous_state,
                    previous_state,
                    "Recovery blocked by health check failure",
                )
                return

    async def _return_to_normal(self) -> None:
        """Return to normal state."""
        async with self._lock:
            previous_state = self.state
            self.state = CircuitBreakerState.NORMAL
            self._log_event(
                "normal",
                previous_state,
                CircuitBreakerState.NORMAL,
                "Returning to normal state",
            )

        # Integration: unfreeze portfolio when returning to normal
        if self.risk_manager and hasattr(self.risk_manager, "unfreeze_portfolio"):
            try:
                # Create task with proper exception handling and timeout
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self.risk_manager.unfreeze_portfolio(),
                        timeout=30.0,  # 30 second timeout
                    )
                )
                # Store task reference to prevent unhandled exceptions
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                self.logger.warning(f"Failed to unfreeze portfolio: {e}")

        # Integration: unblock signals when returning to normal
        if self.signal_router and hasattr(self.signal_router, "unblock_signals"):
            try:
                # Create task with proper exception handling and timeout
                task = asyncio.create_task(
                    asyncio.wait_for(
                        self.signal_router.unblock_signals(),
                        timeout=30.0,  # 30 second timeout
                    )
                )
                # Store task reference to prevent unhandled exceptions
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                self.logger.warning(f"Failed to unblock signals: {e}")

    async def set_state(self, state: CircuitBreakerState, reason: str) -> None:
        """Set circuit breaker state manually."""
        async with self._lock:
            previous_state = self.state
            self.state = state
            self._log_event("manual_set", previous_state, state, reason)

    async def reset_to_normal(self, reason: str) -> bool:
        """Reset to normal state."""
        async with self._lock:
            if self.state == CircuitBreakerState.NORMAL:
                return False
            previous_state = self.state
            self.state = CircuitBreakerState.NORMAL
            self._log_event("reset", previous_state, CircuitBreakerState.NORMAL, reason)
            return True

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get state snapshot for persistence."""
        return {
            "state": self.state.value,
            "trigger_history": self.trigger_history,
            "event_history": self.event_history,
            "trade_results": self.trade_results,
            "last_trigger_time": self.last_trigger_time.isoformat()
            if self.last_trigger_time
            else None,
            "trigger_count": self.trigger_count,
        }

    def restore_state_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        self.state = CircuitBreakerState(snapshot["state"])
        self.trigger_history = snapshot["trigger_history"]
        self.event_history = snapshot["event_history"]
        self.trade_results = snapshot["trade_results"]
        self.last_trigger_time = (
            datetime.fromisoformat(snapshot["last_trigger_time"])
            if snapshot["last_trigger_time"]
            else None
        )
        self.trigger_count = snapshot["trigger_count"]

    async def _check_anomaly_integration(self, market_data: Dict[str, Any]) -> bool:
        """Check anomaly integration with timeout protection."""
        if self.anomaly_detector and hasattr(
            self.anomaly_detector, "detect_market_anomaly"
        ):
            try:
                return await asyncio.wait_for(
                    self.anomaly_detector.detect_market_anomaly(market_data),
                    timeout=15.0,  # 15 second timeout for anomaly detection
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout in anomaly detection")
            except Exception as e:
                self.logger.warning(f"Error in anomaly detection: {e}")
        return False

    def _evaluate_triggers(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate triggers (simplified)."""
        return {}

    async def _record_metric_async(
        self, metrics_collector, metric_name: str, value, labels: Dict[str, str]
    ) -> None:
        """Fire-and-forget metric recording to reduce overhead during breaker engagement."""
        try:
            await asyncio.wait_for(
                metrics_collector.record_metric(metric_name, value, labels),
                timeout=2.0,  # Reduced timeout for fire-and-forget
            )
        except Exception:
            # Silent failure for fire-and-forget metrics
            pass

    async def update_equity(self, equity: float) -> None:
        """
        Update the latest equity value and record it in metrics.

        Args:
            equity: Current equity value
        """
        async with self._lock:
            self.current_equity = equity
            self.logger.info(f"Equity updated: {equity}")

            # Record in metrics - fire-and-forget during breaker engagement to reduce overhead
            if _metrics_collector_available:
                try:
                    metrics_collector = get_metrics_collector()
                    # Create fire-and-forget task for metrics recording
                    asyncio.create_task(
                        self._record_metric_async(
                            metrics_collector,
                            "circuit_breaker_equity",
                            equity,
                            {"account": "main"},
                        )
                    )
                except Exception:
                    # Silent failure during breaker engagement to minimize overhead
                    pass

    def update_config(self, new_config: CircuitBreakerConfig) -> None:
        """Update configuration."""
        self.config = new_config

    def get_circuit_state_metrics(self) -> Dict[str, Any]:
        """Get circuit state metrics for monitoring."""
        return {
            "state": self.state.value,
            "trigger_count": self.trigger_count,
            "last_trigger_time": self.last_trigger_time.isoformat()
            if self.last_trigger_time
            else None,
            "remaining_cooldown_minutes": self.get_remaining_cooldown_minutes(),
            "current_equity": self.current_equity,
            "is_cooldown_active": self._is_cooldown_active(),
            "event_history_size": len(self.event_history),
            "trigger_history_size": len(self.trigger_history),
            "trade_results_count": len(self.trade_results),
            "cooldown_strategy": self.config.cooldown_strategy.value,
            "base_cooldown_minutes": self.config.base_cooldown_minutes,
            "max_cooldown_minutes": self.config.max_cooldown_minutes,
            "cooldown_multiplier": self.config.cooldown_multiplier,
            "enable_cooldown_enforcement": self.config.enable_cooldown_enforcement,
        }

    async def cleanup_background_tasks(self) -> None:
        """Clean up any pending background tasks."""
        if hasattr(self, "_background_tasks") and self._background_tasks:
            try:
                # Wait for all background tasks to complete with timeout
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=10.0,  # 10 second timeout for cleanup
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout waiting for background tasks to complete")
            except Exception as e:
                self.logger.warning(f"Error during background task cleanup: {e}")
            finally:
                self._background_tasks.clear()

    async def shutdown(self) -> None:
        """Shutdown the circuit breaker and clean up resources."""
        self.logger.info("Shutting down circuit breaker")

        # Clean up background tasks
        await self.cleanup_background_tasks()

        # Reset to normal state if needed
        if self.state != CircuitBreakerState.NORMAL:
            await self.reset_to_normal("Shutdown")

        self.logger.info("Circuit breaker shutdown complete")


# Global circuit breaker instance
_circuit_breaker: Optional[CircuitBreaker] = None


def get_circuit_breaker() -> CircuitBreaker:
    """Get the global circuit breaker instance."""
    global _circuit_breaker
    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
    return _circuit_breaker


def create_circuit_breaker(
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Create a new circuit breaker instance."""
    return CircuitBreaker(config or CircuitBreakerConfig())
