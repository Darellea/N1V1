"""
Smart Execution Layer

Unified, adaptive, and resilient execution layer that intelligently selects
and manages execution policies (TWAP, VWAP, DCA, Market/Limit) with validation,
retry/fallback mechanisms, and comprehensive logging.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.contracts import SignalType, TradingSignal
from core.types.order_types import Order, OrderStatus, OrderType
from utils.config_loader import get_config
from utils.logger import get_trade_logger

from .adaptive_pricer import AdaptivePricer
from .dca_executor import DCAExecutor
from .execution_types import ExecutionPolicy, ExecutionStatus
from .smart_order_executor import SmartOrderExecutor
from .twap_executor import TWAPExecutor
from .validator import ExecutionValidator
from .vwap_executor import VWAPExecutor

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


@dataclass
class ExecutionResult:
    """Result of execution operation."""

    execution_id: str
    status: ExecutionStatus
    orders: List[Order]
    policy_used: ExecutionPolicy
    total_amount: Decimal
    executed_amount: Decimal
    average_price: Optional[Decimal]
    total_cost: Decimal
    fees: Decimal
    slippage: Decimal
    duration_ms: int
    retries: int
    fallback_used: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        data["policy_used"] = self.policy_used.value
        return data


class IExecutionStrategy(ABC):
    """Unified interface for all execution strategies."""

    @abstractmethod
    async def prepare_order(
        self, signal: TradingSignal, context: Dict[str, Any]
    ) -> TradingSignal:
        """Prepare order for execution."""
        pass

    @abstractmethod
    async def execute(self) -> List[Order]:
        """Execute the prepared order."""
        pass

    @abstractmethod
    async def monitor(self) -> ExecutionStatus:
        """Monitor execution progress."""
        pass

    @abstractmethod
    async def finalize(self) -> ExecutionResult:
        """Finalize execution and return results."""
        pass

    @abstractmethod
    async def cancel(self) -> bool:
        """Cancel execution."""
        pass


class ExecutionSmartLayer:
    """
    Main coordinator for the Smart Execution Layer.

    Intelligently selects and manages execution policies with validation,
    retry/fallback mechanisms, and comprehensive logging.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Smart Execution Layer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize components
        validation_config = self.config.get("validation", {})
        validation_config["test_mode"] = self.config.get("test_mode", False)
        self.validator = ExecutionValidator(validation_config)
        from core.component_factory import ComponentFactory

        self.retry_manager = ComponentFactory.get("retry_manager")
        self.adaptive_pricer = AdaptivePricer(self.config.get("adaptive_pricing", {}))

        # Initialize executors with test_mode flag
        test_mode = self.config.get("test_mode", False)
        twap_config = self.config.get("twap", {})
        twap_config["test_mode"] = test_mode

        vwap_config = self.config.get("vwap", {})
        vwap_config["test_mode"] = test_mode

        dca_config = self.config.get("dca", {})
        dca_config["test_mode"] = test_mode

        smart_split_config = self.config.get("smart_split", {})
        smart_split_config["test_mode"] = test_mode

        self.executors = {
            ExecutionPolicy.TWAP: TWAPExecutor(twap_config),
            ExecutionPolicy.VWAP: VWAPExecutor(vwap_config),
            ExecutionPolicy.DCA: DCAExecutor(dca_config),
            ExecutionPolicy.SMART_SPLIT: SmartOrderExecutor(smart_split_config),
        }

        # Policy selection thresholds
        self.policy_thresholds = self.config.get(
            "policy_thresholds",
            {
                "large_order": 10000,  # Orders above this use advanced strategies
                "high_spread": 0.005,  # 0.5% spread threshold
                "liquidity_stable": 0.7,  # 70% liquidity stability threshold
            },
        )

        # Active executions
        self.active_executions: Dict[str, Dict[str, Any]] = {}

        self.logger.info("ExecutionSmartLayer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "policy_thresholds": {
                "large_order": 10000,
                "high_spread": 0.005,
                "liquidity_stable": 0.7,
            },
            "validation": {
                "enabled": True,
                "check_balance": True,
                "check_slippage": True,
                "max_slippage_pct": 0.02,
            },
            "retry": {
                "enabled": True,
                "max_retries": 3,
                "backoff_base": 1.0,
                "max_backoff": 30.0,
                "retry_on_errors": ["network", "exchange_timeout", "rate_limit"],
            },
            "adaptive_pricing": {
                "enabled": True,
                "atr_window": 14,
                "price_adjustment_multiplier": 0.5,
                "max_price_adjustment_pct": 0.05,
            },
            "twap": {"duration_minutes": 30, "parts": 10},
            "vwap": {"lookback_minutes": 60, "parts": 10},
            "dca": {"interval_minutes": 60, "parts": 5},
            "smart_split": {
                "split_threshold": 5000,
                "max_parts": 5,
                "delay_seconds": 2.0,
            },
        }

    async def execute_signal(
        self, signal: TradingSignal, market_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute a trading signal using the Smart Execution Layer.

        Args:
            signal: Trading signal to execute
            market_context: Market context information

        Returns:
            ExecutionResult with comprehensive execution details
        """
        execution_id = str(uuid.uuid4())
        start_time = asyncio.get_event_loop().time()

        try:
            self.logger.info(
                f"Starting execution {execution_id} for signal: {signal.symbol} {signal.amount}"
            )

            # Phase 1: Prepare
            prepared_signal = await self._prepare_execution(
                signal, market_context or {}
            )

            # Phase 2: Validate
            if not await self.validator.validate_signal(
                prepared_signal, market_context or {}
            ):
                return ExecutionResult(
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILED,
                    orders=[],
                    policy_used=ExecutionPolicy.MARKET,
                    total_amount=signal.amount,
                    executed_amount=Decimal(0),
                    average_price=None,
                    total_cost=Decimal(0),
                    fees=Decimal(0),
                    slippage=Decimal(0),
                    duration_ms=int(
                        (asyncio.get_event_loop().time() - start_time) * 1000
                    ),
                    retries=0,
                    fallback_used=False,
                    error_message="Signal validation failed",
                )

            # Phase 3: Select Policy
            policy = self._select_execution_policy(
                prepared_signal, market_context or {}
            )

            # Phase 4: Execute with Retry/Fallback
            # Get idempotency key for safe retries
            idempotency_key = getattr(prepared_signal, "idempotency_key", None)
            execution_result = await self.retry_manager.execute_with_retry(
                self._execute_with_policy,
                prepared_signal,
                policy,
                market_context or {},
                allow_side_effect_retry=True,
                idempotency_key=idempotency_key,
                is_side_effect=True,
            )

            # Phase 5: Finalize
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Determine the actual policy used (may be fallback policy)
            final_policy_value = execution_result.get("final_policy", policy.value)
            actual_policy_used = ExecutionPolicy(final_policy_value)

            # Update metadata to indicate fallback if used
            metadata = execution_result.get("metadata", {})
            if execution_result.get("fallback_used", False):
                metadata["fallback_applied"] = True
                metadata["original_policy"] = policy.value
                metadata["fallback_policy"] = final_policy_value

            result = ExecutionResult(
                execution_id=execution_id,
                status=execution_result.get("status", ExecutionStatus.COMPLETED),
                orders=execution_result.get("orders", []),
                policy_used=actual_policy_used,
                total_amount=signal.amount,
                executed_amount=execution_result.get("executed_amount", Decimal(0)),
                average_price=execution_result.get("average_price"),
                total_cost=execution_result.get("total_cost", Decimal(0)),
                fees=execution_result.get("fees", Decimal(0)),
                slippage=execution_result.get("slippage", Decimal(0)),
                duration_ms=duration_ms,
                retries=execution_result.get("retries", 0),
                fallback_used=execution_result.get("fallback_used", False),
                metadata=metadata,
            )

            # Log execution result
            self._log_execution_result(result)

            # Publish event
            await self._publish_execution_event(result)

            return result

        except Exception as e:
            self.logger.error(f"Execution {execution_id} failed: {e}", exc_info=True)

            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return ExecutionResult(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                orders=[],
                policy_used=ExecutionPolicy.MARKET,
                total_amount=signal.amount,
                executed_amount=Decimal(0),
                average_price=None,
                total_cost=Decimal(0),
                fees=Decimal(0),
                slippage=Decimal(0),
                duration_ms=duration_ms,
                retries=0,
                fallback_used=False,
                error_message=str(e),
            )

    async def _prepare_execution(
        self, signal: TradingSignal, context: Dict[str, Any]
    ) -> TradingSignal:
        """
        Prepare signal for execution.

        Args:
            signal: Original trading signal
            context: Market context

        Returns:
            Prepared trading signal
        """
        prepared_signal = signal.copy()

        # Apply adaptive pricing if enabled
        if self.config.get("adaptive_pricing", {}).get("enabled", True):
            adjusted_price = await self.adaptive_pricer.adjust_price(signal, context)
            if adjusted_price:
                prepared_signal.price = adjusted_price
                self.logger.debug(
                    f"Applied adaptive pricing: {signal.price} -> {adjusted_price}"
                )

        # Add execution metadata
        if prepared_signal.metadata is None:
            prepared_signal.metadata = {}

        prepared_signal.metadata.update(
            {
                "execution_prepared": True,
                "original_price": signal.price,
                "adaptive_pricing_applied": prepared_signal.price != signal.price,
            }
        )

        return prepared_signal

    def _select_execution_policy(
        self, signal: TradingSignal, context: Dict[str, Any]
    ) -> ExecutionPolicy:
        """
        Select the appropriate execution policy based on signal and market conditions.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            Selected execution policy
        """
        # Calculate order value
        price = signal.current_price or signal.price or Decimal(1)
        order_value = signal.amount * price

        # Get market conditions
        spread_pct = context.get("spread_pct", 0.002)  # Default 0.2%
        liquidity_stability = context.get("liquidity_stability", 0.8)  # Default 80%

        # Policy selection logic - prioritize market conditions and order size over explicit order type
        if order_value > self.policy_thresholds["large_order"]:
            # Large orders need advanced execution regardless of specified order type
            if liquidity_stability > self.policy_thresholds["liquidity_stable"]:
                return ExecutionPolicy.VWAP  # Good liquidity, use VWAP
            elif spread_pct > self.policy_thresholds["high_spread"]:
                return ExecutionPolicy.DCA  # High spread, use DCA
            else:
                return ExecutionPolicy.TWAP  # Default for large orders

        elif spread_pct > self.policy_thresholds["high_spread"]:
            # High spread conditions
            return ExecutionPolicy.DCA

        elif order_value > self.policy_thresholds["large_order"] * 0.5:
            # Medium-large orders
            return ExecutionPolicy.SMART_SPLIT

        else:
            # Small orders - respect explicit order type if specified
            if signal.order_type == OrderType.LIMIT and signal.price:
                return ExecutionPolicy.LIMIT
            elif signal.order_type == OrderType.MARKET:
                return ExecutionPolicy.MARKET
            else:
                # Default to market for small orders
                return ExecutionPolicy.MARKET

    async def _execute_with_policy(
        self, signal: TradingSignal, policy: ExecutionPolicy, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute order using the selected policy.

        Args:
            signal: Trading signal
            policy: Execution policy
            context: Market context

        Returns:
            Execution results
        """
        if policy in [ExecutionPolicy.MARKET, ExecutionPolicy.LIMIT]:
            # Simple market/limit execution - wrap in a function for retry manager
            async def simple_execution():
                return await self._execute_simple_order(signal, policy)

            return await simple_execution()

        # Use advanced executor
        executor = self.executors.get(policy)
        if not executor:
            self.logger.warning(
                f"Executor not found for policy {policy}, falling back to market"
            )

            async def fallback_execution():
                return await self._execute_simple_order(signal, ExecutionPolicy.MARKET)

            return await fallback_execution()

        # Execute using the selected executor
        orders = await executor.execute_order(signal)

        # Calculate execution metrics
        executed_amount = sum(order.filled for order in orders)
        total_cost = sum(order.cost for order in orders)
        fees = sum(order.fee.get("cost", 0) if order.fee else 0 for order in orders)

        # Calculate average price and slippage
        average_price = None
        slippage = Decimal(0)

        if executed_amount > 0:
            average_price = total_cost / executed_amount

            # Calculate slippage (difference from expected price)
            expected_price = signal.price or signal.current_price or average_price
            if expected_price:
                slippage = abs(average_price - expected_price) / expected_price

        return {
            "status": ExecutionStatus.COMPLETED,
            "orders": orders,
            "executed_amount": executed_amount,
            "average_price": average_price,
            "total_cost": total_cost,
            "fees": fees,
            "slippage": slippage,
            "retries": 0,
            "fallback_used": False,
            "metadata": {"policy": policy.value, "orders_count": len(orders)},
        }

    async def _execute_simple_order(
        self, signal: TradingSignal, policy: ExecutionPolicy
    ) -> Dict[str, Any]:
        """
        Execute simple market or limit order.

        Args:
            signal: Trading signal
            policy: Execution policy (MARKET or LIMIT)

        Returns:
            Execution results
        """
        # Create mock order for demonstration
        # In real implementation, this would call exchange API
        order_id = str(uuid.uuid4())

        # Determine order side
        if signal.signal_type == SignalType.ENTRY_LONG:
            side = "buy"
        elif signal.signal_type == SignalType.ENTRY_SHORT:
            side = "sell"
        elif signal.signal_type == SignalType.EXIT_LONG:
            side = "sell"
        elif signal.signal_type == SignalType.EXIT_SHORT:
            side = "buy"
        else:
            side = "buy"

        # Simulate execution
        executed_price = signal.price or signal.current_price or Decimal(100)
        slippage = Decimal(0.001)  # 0.1% slippage

        if policy == ExecutionPolicy.MARKET:
            # Market order with some slippage
            actual_price = executed_price * (1 + slippage)
        else:
            # Limit order at specified price
            actual_price = signal.price or executed_price

        order = Order(
            id=order_id,
            symbol=signal.symbol,
            type=OrderType.MARKET
            if policy == ExecutionPolicy.MARKET
            else OrderType.LIMIT,
            side=side,
            amount=signal.amount,
            price=actual_price,
            status=OrderStatus.FILLED,
            timestamp=signal.timestamp,
            filled=signal.amount,
            remaining=Decimal(0),
            cost=signal.amount * actual_price,
            fee={
                "cost": signal.amount * actual_price * Decimal(0.001),
                "currency": "USD",
            },
        )

        return {
            "status": ExecutionStatus.COMPLETED,
            "orders": [order],
            "executed_amount": signal.amount,
            "average_price": actual_price,
            "total_cost": signal.amount * actual_price,
            "fees": signal.amount * actual_price * Decimal(0.001),
            "slippage": slippage,
            "retries": 0,
            "fallback_used": False,
            "metadata": {"policy": policy.value, "simple_execution": True},
        }

    def _log_execution_result(self, result: ExecutionResult) -> None:
        """Log execution result."""
        self.logger.info(
            f"Execution {result.execution_id} completed: "
            f"policy={result.policy_used.value}, "
            f"amount={result.executed_amount}/{result.total_amount}, "
            f"avg_price={result.average_price}, "
            f"slippage={result.slippage:.4f}, "
            f"duration={result.duration_ms}ms, "
            f"retries={result.retries}"
        )

        # Log to trade logger
        trade_logger.performance("Execution Completed", result.to_dict())

    async def _publish_execution_event(self, result: ExecutionResult) -> None:
        """Publish execution event to event bus."""
        # This would publish to an event bus system
        # For now, just log the event
        event_data = {
            "event_type": "EXECUTION_COMPLETED",
            "execution_id": result.execution_id,
            "status": result.status.value,
            "policy": result.policy_used.value,
            "executed_amount": float(result.executed_amount),
            "total_amount": float(result.total_amount),
            "average_price": float(result.average_price)
            if result.average_price
            else None,
            "slippage": float(result.slippage),
            "duration_ms": result.duration_ms,
            "retries": result.retries,
            "fallback_used": result.fallback_used,
        }

        self.logger.debug(f"Published execution event: {event_data}")

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.

        Args:
            execution_id: Execution ID to cancel

        Returns:
            True if cancellation was successful
        """
        execution = self.active_executions.get(execution_id)
        if not execution:
            self.logger.warning(f"Execution {execution_id} not found")
            return False

        # Cancel the execution
        # Implementation would depend on the specific executor
        self.logger.info(f"Cancelled execution {execution_id}")
        return True

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Execution status information
        """
        execution = self.active_executions.get(execution_id)
        if not execution:
            return None

        return {
            "execution_id": execution_id,
            "status": execution.get("status", "unknown"),
            "policy": execution.get("policy", "unknown"),
            "progress": execution.get("progress", 0.0),
            "executed_amount": execution.get("executed_amount", 0),
            "total_amount": execution.get("total_amount", 0),
        }

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """
        Get all active executions.

        Returns:
            List of active execution information
        """
        return [
            {
                "execution_id": exec_id,
                "status": execution.get("status", "unknown"),
                "policy": execution.get("policy", "unknown"),
                "progress": execution.get("progress", 0.0),
            }
            for exec_id, execution in self.active_executions.items()
        ]


# Global instance
_execution_layer: Optional[ExecutionSmartLayer] = None


def get_execution_layer() -> ExecutionSmartLayer:
    """Get the global execution layer instance."""
    global _execution_layer
    if _execution_layer is None:
        config = get_config("execution_layer", {})
        _execution_layer = ExecutionSmartLayer(config)
    return _execution_layer


async def execute_signal(
    signal: TradingSignal, market_context: Optional[Dict[str, Any]] = None
) -> ExecutionResult:
    """
    Convenience function to execute a signal using the smart layer.

    Args:
        signal: Trading signal to execute
        market_context: Market context information

    Returns:
        ExecutionResult with comprehensive execution details
    """
    layer = get_execution_layer()
    return await layer.execute_signal(signal, market_context)
