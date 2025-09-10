"""
Strategy Ensemble Manager for multi-strategy portfolio management.

Provides comprehensive ensemble management capabilities including dynamic capital allocation,
performance aggregation, and coordinated execution across multiple strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from decimal import Decimal
import statistics
import uuid

from core.contracts import TradingSignal
from core.signal_router.events import (
    StrategySwitchEvent,
    create_strategy_switch_event,
    EventType
)
from core.signal_router.event_bus import get_default_enhanced_event_bus
from utils.logger import get_trade_logger
# Import metrics functions or implement locally

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Represents capital allocation for a strategy."""
    strategy_id: str
    weight: float  # 0.0 to 1.0
    capital_allocated: Decimal
    last_updated: datetime = field(default_factory=datetime.now)
    performance_score: float = 0.0


@dataclass
class EnsemblePerformance:
    """Aggregated performance metrics for the strategy ensemble."""
    total_capital: Decimal
    total_pnl: Decimal = Decimal('0')
    total_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    strategy_contributions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class EnsembleSignal:
    """Signal from the ensemble with allocation context."""
    original_signal: TradingSignal
    strategy_id: str
    allocated_weight: float
    allocated_quantity: Decimal
    ensemble_context: Dict[str, Any]


class StrategyEnsembleManager:
    """
    Manages a portfolio of trading strategies with dynamic capital allocation.

    Features:
    - Dynamic capital allocation based on performance metrics
    - Coordinated execution across multiple strategies
    - Performance aggregation and reporting
    - Risk management at portfolio level
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy ensemble manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.event_bus = get_default_enhanced_event_bus()
        self.logger = get_trade_logger()

        # Configuration with defaults
        self.rebalance_interval_sec = self.config.get('rebalance_interval_sec', 3600)  # 1 hour
        self.allocation_method = self.config.get('allocation_method', 'sharpe_weighted')
        self.min_weight = self.config.get('min_weight', 0.05)  # 5%
        self.max_weight = self.config.get('max_weight', 0.4)   # 40%
        self.portfolio_risk_limit = self.config.get('portfolio_risk_limit', 0.1)  # 10%
        self.performance_window_days = self.config.get('performance_window_days', 30)

        # Core state
        self.total_capital = Decimal(str(self.config.get('total_capital', '10000')))
        self.strategies: Dict[str, Any] = {}  # strategy_id -> strategy_instance
        self.allocations: Dict[str, StrategyAllocation] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.ensemble_performance = EnsemblePerformance(total_capital=self.total_capital)

        # Control flags
        self._running = False
        self._rebalance_task: Optional[asyncio.Task] = None

        # Risk management
        self.portfolio_exposure: Dict[str, Decimal] = {}  # symbol -> exposure
        self.strategy_positions: Dict[str, Dict[str, Decimal]] = {}  # strategy_id -> {symbol: position}

        logger.info("StrategyEnsembleManager initialized")

    async def start(self) -> None:
        """Start the ensemble manager."""
        if self._running:
            return

        self._running = True
        self._rebalance_task = asyncio.create_task(self._rebalance_loop())

        # Initialize allocations if strategies exist
        if self.strategies:
            await self._initialize_allocations()

        logger.info("StrategyEnsembleManager started")

    async def stop(self) -> None:
        """Stop the ensemble manager."""
        if not self._running:
            return

        self._running = False

        if self._rebalance_task:
            self._rebalance_task.cancel()
            try:
                await self._rebalance_task
            except asyncio.CancelledError:
                pass

        logger.info("StrategyEnsembleManager stopped")

    async def add_strategy(self, strategy_id: str, strategy_instance: Any,
                          initial_weight: Optional[float] = None) -> None:
        """
        Add a strategy to the ensemble.

        Args:
            strategy_id: Unique identifier for the strategy
            strategy_instance: Strategy instance
            initial_weight: Initial allocation weight (optional)
        """
        if strategy_id in self.strategies:
            logger.warning(f"Strategy {strategy_id} already exists in ensemble")
            return

        self.strategies[strategy_id] = strategy_instance
        self.performance_history[strategy_id] = []
        self.strategy_positions[strategy_id] = {}

        # Set initial allocation
        if initial_weight is None:
            initial_weight = 1.0 / len(self.strategies)  # Equal weight

        allocation = StrategyAllocation(
            strategy_id=strategy_id,
            weight=initial_weight,
            capital_allocated=self.total_capital * Decimal(str(initial_weight))
        )
        self.allocations[strategy_id] = allocation

        # Normalize weights after adding strategy
        await self._normalize_allocations()

        # Publish strategy addition event
        await self._publish_strategy_event(
            strategy_id=strategy_id,
            event_type="strategy_added",
            weight=self.allocations[strategy_id].weight,  # Use normalized weight
            rationale=f"Strategy {strategy_id} added to ensemble"
        )

        logger.info(f"Added strategy {strategy_id} with normalized weight {self.allocations[strategy_id].weight:.3f}")

    async def remove_strategy(self, strategy_id: str) -> None:
        """
        Remove a strategy from the ensemble.

        Args:
            strategy_id: Strategy to remove
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found in ensemble")
            return

        # Close any open positions for this strategy
        await self._close_strategy_positions(strategy_id)

        # Remove strategy and its data
        del self.strategies[strategy_id]
        del self.allocations[strategy_id]
        del self.performance_history[strategy_id]
        del self.strategy_positions[strategy_id]

        # Rebalance remaining strategies
        if self.strategies:
            await self._rebalance_allocations()

        # Publish strategy removal event
        await self._publish_strategy_event(
            strategy_id=strategy_id,
            event_type="strategy_removed",
            weight=0.0,
            rationale=f"Strategy {strategy_id} removed from ensemble"
        )

        logger.info(f"Removed strategy {strategy_id}")

    async def route_signal(self, signal: TradingSignal, strategy_id: str,
                          market_data: Optional[Dict[str, Any]] = None) -> Optional[EnsembleSignal]:
        """
        Route a signal from a strategy through the ensemble.

        Args:
            signal: Trading signal from strategy
            strategy_id: ID of the strategy that generated the signal
            market_data: Current market data

        Returns:
            EnsembleSignal with allocation context, or None if rejected
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Signal from unknown strategy {strategy_id}")
            return None

        # Check if strategy has allocation
        if strategy_id not in self.allocations:
            logger.warning(f"No allocation for strategy {strategy_id}")
            return None

        allocation = self.allocations[strategy_id]

        # Calculate position size based on allocation
        allocated_quantity = self._calculate_allocated_quantity(signal, allocation)

        # Check portfolio risk limits
        if not await self._check_portfolio_risk(signal, allocated_quantity):
            logger.warning(f"Signal rejected due to portfolio risk limits: {signal.symbol}")
            return None

        # Create ensemble signal
        ensemble_signal = EnsembleSignal(
            original_signal=signal,
            strategy_id=strategy_id,
            allocated_weight=allocation.weight,
            allocated_quantity=allocated_quantity,
            ensemble_context={
                "allocation_weight": allocation.weight,
                "capital_allocated": float(allocation.capital_allocated),
                "portfolio_exposure": float(self._get_portfolio_exposure(signal.symbol)),
                "strategy_performance": allocation.performance_score
            }
        )

        # Update position tracking
        await self._update_position_tracking(strategy_id, signal, allocated_quantity)

        logger.info(
            f"Signal routed: {strategy_id} -> {signal.symbol} {signal.side} "
            f"quantity={allocated_quantity} (weight={allocation.weight:.3f})"
        )

        return ensemble_signal

    async def update_performance(self, strategy_id: str, performance_data: Dict[str, Any]) -> None:
        """
        Update performance data for a strategy.

        Args:
            performance_data: Performance metrics from the strategy
        """
        if strategy_id not in self.performance_history:
            logger.warning(f"No performance history for strategy {strategy_id}")
            return

        # Add timestamp if not present
        if 'timestamp' not in performance_data:
            performance_data['timestamp'] = datetime.now()

        # Store performance data
        self.performance_history[strategy_id].append(performance_data)

        # Keep only recent performance data
        cutoff_date = datetime.now() - timedelta(days=self.performance_window_days)
        self.performance_history[strategy_id] = [
            p for p in self.performance_history[strategy_id]
            if p['timestamp'] > cutoff_date
        ]

        # Update strategy allocation performance score
        if strategy_id in self.allocations:
            self.allocations[strategy_id].performance_score = self._calculate_performance_score(strategy_id)

        # Update ensemble performance
        await self._update_ensemble_performance()

        logger.debug(f"Updated performance for strategy {strategy_id}")

    async def get_portfolio_performance(self) -> EnsemblePerformance:
        """Get current portfolio performance metrics."""
        await self._update_ensemble_performance()
        return self.ensemble_performance

    def get_strategy_allocations(self) -> Dict[str, Dict[str, Any]]:
        """Get current strategy allocations."""
        return {
            strategy_id: {
                "weight": allocation.weight,
                "capital_allocated": float(allocation.capital_allocated),
                "performance_score": allocation.performance_score,
                "last_updated": allocation.last_updated.isoformat()
            }
            for strategy_id, allocation in self.allocations.items()
        }

    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get comprehensive ensemble status."""
        return {
            "total_capital": float(self.total_capital),
            "active_strategies": len(self.strategies),
            "total_allocations": len(self.allocations),
            "portfolio_exposure": {symbol: float(exposure) for symbol, exposure in self.portfolio_exposure.items()},
            "allocation_method": self.allocation_method,
            "rebalance_interval_sec": self.rebalance_interval_sec,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "portfolio_risk_limit": self.portfolio_risk_limit,
            "running": self._running
        }

    async def _initialize_allocations(self) -> None:
        """Initialize allocations for existing strategies."""
        if not self.strategies:
            return

        # Start with equal weights
        equal_weight = 1.0 / len(self.strategies)

        for strategy_id in self.strategies.keys():
            allocation = StrategyAllocation(
                strategy_id=strategy_id,
                weight=equal_weight,
                capital_allocated=self.total_capital * Decimal(str(equal_weight))
            )
            self.allocations[strategy_id] = allocation

        logger.info(f"Initialized allocations for {len(self.strategies)} strategies")

    async def _rebalance_loop(self) -> None:
        """Main rebalancing loop."""
        while self._running:
            try:
                await asyncio.sleep(self.rebalance_interval_sec)
                if self.strategies:
                    await self._rebalance_allocations()
            except Exception as e:
                logger.exception(f"Error in rebalance loop: {e}")
                await asyncio.sleep(10)

    async def _rebalance_allocations(self) -> None:
        """Rebalance strategy allocations based on performance."""
        if not self.strategies:
            return

        # Calculate new weights based on allocation method
        if self.allocation_method == 'sharpe_weighted':
            new_weights = await self._calculate_sharpe_weights()
        elif self.allocation_method == 'equal_weight':
            new_weights = await self._calculate_equal_weights()
        else:
            logger.warning(f"Unknown allocation method: {self.allocation_method}")
            return

        # Apply min/max weight constraints
        new_weights = self._apply_weight_constraints(new_weights)

        # Update allocations
        weight_changes = []
        for strategy_id, new_weight in new_weights.items():
            if strategy_id in self.allocations:
                old_weight = self.allocations[strategy_id].weight
                if abs(old_weight - new_weight) > 0.01:  # Significant change
                    weight_changes.append((strategy_id, old_weight, new_weight))

                self.allocations[strategy_id].weight = new_weight
                self.allocations[strategy_id].capital_allocated = self.total_capital * Decimal(str(new_weight))
                self.allocations[strategy_id].last_updated = datetime.now()

        # Log significant changes
        if weight_changes:
            for strategy_id, old_weight, new_weight in weight_changes:
                logger.info(
                    f"Allocation updated: {strategy_id} "
                    f"{old_weight:.3f} -> {new_weight:.3f}"
                )

                # Publish allocation update event
                await self._publish_allocation_event(strategy_id, new_weight)

        logger.info("Allocation rebalancing completed")

    async def _calculate_sharpe_weights(self) -> Dict[str, float]:
        """Calculate weights based on Sharpe ratios."""
        weights = {}
        total_score = 0.0

        for strategy_id in self.strategies.keys():
            score = self._calculate_performance_score(strategy_id)
            weights[strategy_id] = max(0.01, score)  # Minimum weight
            total_score += weights[strategy_id]

        # Normalize to sum to 1.0
        if total_score > 0:
            for strategy_id in weights:
                weights[strategy_id] /= total_score
        else:
            # Fallback to equal weights
            equal_weight = 1.0 / len(self.strategies)
            for strategy_id in weights:
                weights[strategy_id] = equal_weight

        return weights

    async def _calculate_equal_weights(self) -> Dict[str, float]:
        """Calculate equal weights for all strategies."""
        if not self.strategies:
            return {}

        equal_weight = 1.0 / len(self.strategies)
        return {strategy_id: equal_weight for strategy_id in self.strategies.keys()}

    async def _normalize_allocations(self) -> None:
        """Normalize strategy allocations to ensure they sum to 1.0."""
        if not self.allocations:
            return

        # Apply min/max weight constraints first
        constrained_weights = {}
        for strategy_id, allocation in self.allocations.items():
            constrained_weights[strategy_id] = max(self.min_weight, min(self.max_weight, allocation.weight))

        # Renormalize to ensure sum equals 1.0
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for strategy_id in constrained_weights:
                normalized_weight = constrained_weights[strategy_id] / total_weight
                self.allocations[strategy_id].weight = normalized_weight
                self.allocations[strategy_id].capital_allocated = self.total_capital * Decimal(str(normalized_weight))
                self.allocations[strategy_id].last_updated = datetime.now()

        logger.debug(f"Normalized allocations for {len(self.allocations)} strategies")

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply min/max weight constraints."""
        constrained_weights = {}

        for strategy_id, weight in weights.items():
            constrained_weights[strategy_id] = max(self.min_weight, min(self.max_weight, weight))

        # Renormalize after constraints
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            for strategy_id in constrained_weights:
                constrained_weights[strategy_id] /= total_weight

        return constrained_weights

    def _calculate_performance_score(self, strategy_id: str) -> float:
        """Calculate performance score for a strategy."""
        if strategy_id not in self.performance_history:
            return 0.5  # Neutral score

        history = self.performance_history[strategy_id]
        if len(history) < 2:  # Need minimum data
            return 0.5

        try:
            # Extract returns and PnL
            returns = [p.get('daily_return', 0.0) for p in history if 'daily_return' in p]
            pnl_values = [p.get('pnl', 0.0) for p in history if 'pnl' in p]

            if len(returns) < 2:
                return 0.5

            # Calculate multiple performance metrics
            avg_return = statistics.mean(returns)
            total_pnl = sum(pnl_values) if pnl_values else 0.0

            # Calculate Sharpe ratio (simplified)
            sharpe_score = 0.5  # Neutral
            if len(returns) > 1:
                std_return = statistics.stdev(returns)
                if std_return > 0:
                    sharpe = avg_return / std_return
                    # Convert Sharpe to 0-1 scale (assuming Sharpe range of -2 to +2)
                    sharpe_score = max(0.1, min(0.9, 0.5 + sharpe * 0.2))

            # Calculate PnL score (0-1 scale based on total PnL)
            pnl_score = 0.5  # Neutral
            if total_pnl != 0:
                # Normalize PnL to a reasonable range (assuming |PnL| up to 1000)
                pnl_normalized = max(-1000, min(1000, total_pnl))
                pnl_score = max(0.1, min(0.9, 0.5 + (pnl_normalized / 1000) * 0.4))

            # Combine scores with weights
            combined_score = (sharpe_score * 0.6) + (pnl_score * 0.4)

            return max(0.1, min(0.9, combined_score))

        except Exception as e:
            logger.warning(f"Error calculating performance score for {strategy_id}: {e}")

        return 0.5  # Default neutral score

    def _calculate_allocated_quantity(self, signal: TradingSignal, allocation: StrategyAllocation) -> Decimal:
        """Calculate quantity to allocate based on signal and strategy allocation."""
        # This is a simplified calculation - in practice, this would consider:
        # - Current portfolio exposure
        # - Risk limits
        # - Position sizing rules
        # - Market conditions

        base_quantity = Decimal(str(signal.quantity))

        # Scale by allocation weight
        allocated_quantity = base_quantity * Decimal(str(allocation.weight))

        # Apply risk scaling (simplified)
        risk_factor = allocation.performance_score * 2.0  # Bettr performance = higher risk tolerance
        allocated_quantity = allocated_quantity * Decimal(str(risk_factor))

        return allocated_quantity

    async def _check_portfolio_risk(self, signal: TradingSignal, quantity: Decimal) -> bool:
        """Check if the signal respects portfolio risk limits."""
        symbol = signal.symbol

        # Calculate potential new exposure
        current_exposure = self._get_portfolio_exposure(symbol)
        signal_value = quantity * Decimal(str(signal.price))

        if signal.side.lower() == 'buy':
            new_exposure = current_exposure + signal_value
        else:  # sell
            new_exposure = current_exposure - signal_value

        # Check against portfolio risk limit
        max_allowed_exposure = self.total_capital * Decimal(str(self.portfolio_risk_limit))

        if abs(new_exposure) > max_allowed_exposure:
            return False

        return True

    def _get_portfolio_exposure(self, symbol: str) -> Decimal:
        """Get current portfolio exposure for a symbol."""
        return self.portfolio_exposure.get(symbol, Decimal('0'))

    async def _update_position_tracking(self, strategy_id: str, signal: TradingSignal, quantity: Decimal) -> None:
        """Update position tracking for strategy and portfolio."""
        symbol = signal.symbol
        signal_value = quantity * Decimal(str(signal.price))

        # Update strategy positions
        if strategy_id not in self.strategy_positions:
            self.strategy_positions[strategy_id] = {}

        current_position = self.strategy_positions[strategy_id].get(symbol, Decimal('0'))

        if signal.side.lower() == 'buy':
            self.strategy_positions[strategy_id][symbol] = current_position + quantity
        else:  # sell
            self.strategy_positions[strategy_id][symbol] = current_position - quantity

        # Update portfolio exposure
        current_exposure = self.portfolio_exposure.get(symbol, Decimal('0'))

        if signal.side.lower() == 'buy':
            self.portfolio_exposure[symbol] = current_exposure + signal_value
        else:  # sell
            self.portfolio_exposure[symbol] = current_exposure - signal_value

    async def _close_strategy_positions(self, strategy_id: str) -> None:
        """Close all positions for a strategy."""
        if strategy_id not in self.strategy_positions:
            return

        # In a real implementation, this would generate closing orders
        # For now, just clear the tracking
        positions = self.strategy_positions[strategy_id]
        for symbol, quantity in positions.items():
            if symbol in self.portfolio_exposure:
                # Remove this strategy's contribution from portfolio exposure
                position_value = quantity * Decimal('0')  # Assume closed at current price
                self.portfolio_exposure[symbol] -= position_value

        self.strategy_positions[strategy_id].clear()

        logger.info(f"Closed all positions for strategy {strategy_id}")

    async def _update_ensemble_performance(self) -> None:
        """Update ensemble performance metrics."""
        try:
            total_pnl = Decimal('0')
            total_trades = 0
            winning_trades = 0

            # Aggregate performance across all strategies
            for strategy_id, history in self.performance_history.items():
                for record in history:
                    pnl = record.get('pnl', 0)
                    total_pnl += Decimal(str(pnl))
                    total_trades += record.get('trades', 0)

                    if pnl > 0:
                        winning_trades += 1

            # Calculate win rate
            if total_trades > 0:
                self.ensemble_performance.win_rate = winning_trades / total_trades

            self.ensemble_performance.total_pnl = total_pnl
            self.ensemble_performance.total_trades = total_trades
            self.ensemble_performance.last_updated = datetime.now()

            # Calculate risk metrics (simplified)
            # In a real implementation, this would use more sophisticated calculations
            self.ensemble_performance.sharpe_ratio = 1.5  # Placeholder
            self.ensemble_performance.sortino_ratio = 1.8  # Placeholder
            self.ensemble_performance.max_drawdown = 0.05  # Placeholder
            self.ensemble_performance.calmar_ratio = 2.1   # Placeholder

        except Exception as e:
            logger.exception(f"Error updating ensemble performance: {e}")

    async def _publish_strategy_event(self, strategy_id: str, event_type: str,
                                     weight: float, rationale: str) -> None:
        """Publish strategy-related events."""
        event = create_strategy_switch_event(
            previous_strategy=None,  # Not applicable for ensemble events
            new_strategy=strategy_id,
            rationale=rationale,
            confidence=weight,
            market_conditions={"ensemble_event": event_type}
        )

        await self.event_bus.publish_event(event)

    async def _publish_allocation_event(self, strategy_id: str, new_weight: float) -> None:
        """Publish allocation update events."""
        # Create a custom event for allocation updates
        allocation_event = {
            "event_type": "allocation_update",
            "source": "strategy_ensemble",
            "timestamp": datetime.now(),
            "payload": {
                "strategy_id": strategy_id,
                "new_weight": new_weight,
                "capital_allocated": float(self.total_capital * Decimal(str(new_weight)))
            },
            "metadata": {
                "ensemble_total_capital": float(self.total_capital),
                "allocation_method": self.allocation_method
            }
        }

        # For now, log the allocation update
        # In a full implementation, this would be a proper event type
        self.logger.info(
            f"Allocation updated: {strategy_id} = {new_weight:.3f}",
            extra={"allocation_data": allocation_event}
        )


# Global ensemble manager instance
_global_ensemble_manager: Optional[StrategyEnsembleManager] = None


def get_ensemble_manager() -> StrategyEnsembleManager:
    """Get the global ensemble manager instance."""
    global _global_ensemble_manager
    if _global_ensemble_manager is None:
        _global_ensemble_manager = StrategyEnsembleManager()
    return _global_ensemble_manager


def create_ensemble_manager(config: Optional[Dict[str, Any]] = None) -> StrategyEnsembleManager:
    """Create a new ensemble manager instance."""
    return StrategyEnsembleManager(config)
