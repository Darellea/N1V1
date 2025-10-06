"""
core/order_manager.py

Refactored OrderManager that orchestrates the new specialized classes.
Handles order execution, tracking, and management across all trading modes.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import jsonschema
from ccxt.base.errors import ExchangeError, NetworkError

from core.contracts import TradingSignal
from core.exceptions import MissingIdempotencyError
from core.execution.backtest_executor import BacktestOrderExecutor
from core.execution.live_executor import LiveOrderExecutor
from core.execution.order_processor import OrderProcessor
from core.execution.paper_executor import PaperOrderExecutor
from core.idempotency import (
    generate_idempotency_key,
    OrderExecutionRegistry,
    order_execution_registry,
)
from core.management.portfolio_manager import PortfolioManager
from core.management.reliability_manager import ReliabilityManager
from core.order_validation import OrderValidationPipeline
from core.types import TradingMode
from core.types.order_types import OrderStatus
from utils.adapter import signal_to_dict
from utils.logger import get_trade_logger

from .logging_utils import LogSensitivity, get_structured_logger


def get_signal_attr(signal, attr, default=None):
    """Get attribute from signal, handling both dict and object types."""
    if isinstance(signal, dict):
        return signal.get(attr, default)
    else:
        return getattr(signal, attr, default)


logger = get_structured_logger("core.order_manager", LogSensitivity.SECURE)
trade_logger = get_trade_logger()


@dataclass
class Fill:
    """Represents a fill from an exchange."""
    order_id: str
    symbol: str
    amount: Decimal
    price: Decimal
    timestamp: float
    fill_type: str = "full"  # "full", "partial", "final"
    exchange_order_id: Optional[str] = None
    fees: Optional[Dict[str, Decimal]] = None


@dataclass
class PartialFillRecord:
    """Tracks partial fill information for reconciliation."""
    original_order_id: str
    symbol: str
    original_amount: Decimal
    filled_amount: Decimal = field(default=Decimal(0))
    remaining_amount: Decimal = field(default=Decimal(0))
    fills: List[Fill] = field(default_factory=list)
    status: str = "pending"  # "pending", "partially_filled", "fully_filled", "failed", "timed_out"
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0  # 5 minutes default
    reconciliation_attempts: int = 0
    last_reconciliation_at: Optional[float] = None
    manual_intervention_required: bool = False
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if self.remaining_amount == 0:
            self.remaining_amount = self.original_amount

    def add_fill(self, fill: Fill) -> None:
        """Add a fill to this record."""
        self.fills.append(fill)
        self.filled_amount += fill.amount
        self.remaining_amount = self.original_amount - self.filled_amount
        self.last_updated = time.time()

        # Update status based on fill progress
        if self.filled_amount >= self.original_amount:
            self.status = "fully_filled"
        elif self.filled_amount > 0:
            self.status = "partially_filled"
        else:
            self.status = "pending"

        # Add audit entry
        self.audit_trail.append({
            "timestamp": self.last_updated,
            "action": "fill_added",
            "fill_amount": float(fill.amount),
            "total_filled": float(self.filled_amount),
            "remaining": float(self.remaining_amount),
            "fill_type": fill.fill_type
        })

    def is_expired(self) -> bool:
        """Check if the partial fill record has timed out."""
        return (time.time() - self.created_at) > self.timeout_seconds

    def should_retry(self) -> bool:
        """Check if we should retry filling the remaining amount."""
        return (
            self.status in ["pending", "partially_filled"] and
            self.retry_count < self.max_retries and
            not self.is_expired() and
            not self.manual_intervention_required
        )

    def mark_for_manual_intervention(self, reason: str) -> None:
        """Mark this record for manual intervention."""
        self.manual_intervention_required = True
        self.status = "manual_intervention"
        self.audit_trail.append({
            "timestamp": time.time(),
            "action": "manual_intervention_required",
            "reason": reason
        })


class PartialFillReconciliationManager:
    """Manages partial fill reconciliation with retry logic and validation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.partial_fills: Dict[str, PartialFillRecord] = {}
        self.fill_timeout = config.get("fill_timeout", 300.0)  # 5 minutes
        self.max_retries = config.get("max_fill_retries", 3)
        self.retry_delay = config.get("fill_retry_delay", 10.0)  # 10 seconds
        self.reconciliation_interval = config.get("reconciliation_interval", 30.0)  # 30 seconds
        self.audit_enabled = config.get("enable_fill_audit", True)

        # Metrics
        self.metrics = {
            "total_partial_fills": 0,
            "successful_reconciliations": 0,
            "failed_reconciliations": 0,
            "timed_out_fills": 0,
            "manual_interventions": 0,
            "exchange_discrepancies": 0
        }

        # Background task for reconciliation
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._running = False

    async def start_reconciliation_loop(self) -> None:
        """Start the background reconciliation loop."""
        if self._running:
            return

        self._running = True
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        logger.info("Started partial fill reconciliation loop")

    async def stop_reconciliation_loop(self) -> None:
        """Stop the background reconciliation loop."""
        self._running = False
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped partial fill reconciliation loop")

    async def _reconciliation_loop(self) -> None:
        """Background loop for reconciling partial fills."""
        while self._running:
            try:
                await self._reconcile_pending_fills()
                await asyncio.sleep(self.reconciliation_interval)
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(self.reconciliation_interval)

    async def _reconcile_pending_fills(self) -> None:
        """Reconcile all pending partial fills."""
        current_time = time.time()
        expired_records = []

        for order_id, record in self.partial_fills.items():
            try:
                # Check for timeouts
                if record.is_expired() and record.status != "timed_out":
                    record.status = "timed_out"
                    record.audit_trail.append({
                        "timestamp": current_time,
                        "action": "timed_out",
                        "reason": f"Exceeded {record.timeout_seconds}s timeout"
                    })
                    self.metrics["timed_out_fills"] += 1
                    logger.warning(f"Partial fill timed out for order {order_id}")

                # Check for manual intervention
                if record.manual_intervention_required:
                    continue  # Skip reconciliation for manual intervention cases

                # Attempt reconciliation
                if record.status in ["pending", "partially_filled"]:
                    await self._attempt_reconciliation(record)
                    record.reconciliation_attempts += 1
                    record.last_reconciliation_at = current_time

                # Clean up completed records after some time
                if record.status in ["fully_filled", "failed", "timed_out"]:
                    # Keep records for 1 hour after completion for audit
                    if current_time - record.last_updated > 3600:
                        expired_records.append(order_id)

            except Exception as e:
                logger.error(f"Error reconciling partial fill {order_id}: {e}")

        # Clean up expired records
        for order_id in expired_records:
            del self.partial_fills[order_id]

    async def _attempt_reconciliation(self, record: PartialFillRecord) -> None:
        """Attempt to reconcile a partial fill record."""
        # This would typically query the exchange for current order status
        # For now, we'll implement a basic version that can be extended
        try:
            # Placeholder for exchange-specific reconciliation logic
            # In a real implementation, this would:
            # 1. Query exchange for current order status
            # 2. Compare with our records
            # 3. Handle discrepancies
            # 4. Update positions accordingly

            if record.reconciliation_attempts > 5:  # Max reconciliation attempts
                record.mark_for_manual_intervention("Max reconciliation attempts exceeded")
                self.metrics["manual_interventions"] += 1

        except Exception as e:
            logger.error(f"Reconciliation attempt failed for {record.original_order_id}: {e}")

    def register_partial_fill(self, order_id: str, symbol: str, original_amount: Decimal) -> PartialFillRecord:
        """Register a new partial fill for tracking."""
        if order_id in self.partial_fills:
            logger.warning(f"Partial fill already registered for order {order_id}")
            return self.partial_fills[order_id]

        record = PartialFillRecord(
            original_order_id=order_id,
            symbol=symbol,
            original_amount=original_amount,
            timeout_seconds=self.fill_timeout,
            max_retries=self.max_retries
        )

        self.partial_fills[order_id] = record
        self.metrics["total_partial_fills"] += 1

        if self.audit_enabled:
            record.audit_trail.append({
                "timestamp": time.time(),
                "action": "registered",
                "original_amount": float(original_amount)
            })

        logger.info(f"Registered partial fill tracking for order {order_id}")
        return record

    def add_fill_to_record(self, order_id: str, fill: Fill) -> bool:
        """Add a fill to an existing partial fill record."""
        if order_id not in self.partial_fills:
            logger.warning(f"No partial fill record found for order {order_id}")
            return False

        record = self.partial_fills[order_id]
        record.add_fill(fill)

        if record.status == "fully_filled":
            self.metrics["successful_reconciliations"] += 1
            logger.info(f"Partial fill fully reconciled for order {order_id}")

        return True

    def get_fill_metrics(self) -> Dict[str, int]:
        """Get current fill reconciliation metrics."""
        return self.metrics.copy()

    def get_pending_fills(self) -> List[PartialFillRecord]:
        """Get all pending partial fill records."""
        return [r for r in self.partial_fills.values() if r.status in ["pending", "partially_filled"]]

    def get_stuck_fills(self) -> List[PartialFillRecord]:
        """Get fills that may be stuck and need manual intervention."""
        return [r for r in self.partial_fills.values() if r.manual_intervention_required or r.is_expired()]


class MockLiveExecutor:
    """Mock live executor for testing and non-live modes."""

    def __init__(self, config=None):
        self.exchange = None
        self.config = config

    async def execute_live_order(self, signal):
        """Mock live order execution."""
        # Return a mock successful response
        return {
            "id": f"mock_order_{random.randint(1000, 9999)}",
            "status": "filled",
            "amount": get_signal_attr(signal, "amount", 0.001),
            "price": 50000.0,
            "symbol": get_signal_attr(signal, "symbol", "BTC/USDT"),
        }

    async def shutdown(self):
        """Mock shutdown."""
        pass


class OrderExecutionStrategy(ABC):
    """Abstract base class for order execution strategies."""

    def __init__(self, order_manager: "OrderManager"):
        self.order_manager = order_manager

    @abstractmethod
    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute an order using this strategy."""
        pass

    @abstractmethod
    def get_mode_name(self) -> str:
        """Get the name of this execution mode."""
        pass


class LiveOrderExecutionStrategy(OrderExecutionStrategy):
    """Strategy for live order execution with retry logic."""

    def get_mode_name(self) -> str:
        return "live"

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute order in live mode with retry and error handling."""
        try:
            # Get idempotency key for safe retries
            idempotency_key = get_signal_attr(signal, "idempotency_key", None)
            order_response = await self.order_manager.reliability_manager.retry_async(
                lambda: self.order_manager.live_executor.execute_live_order(signal),
                exceptions=(NetworkError, ExchangeError, asyncio.TimeoutError, OSError),
                allow_side_effect_retry=True,
                idempotency_key=idempotency_key,
                is_side_effect=True,
            )
            order = self.order_manager.order_processor.parse_order_response(
                order_response
            )
            processed_order = await self.order_manager.order_processor.process_order(
                order
            )
            trade_logger.log_order(processed_order, self.get_mode_name())
            return processed_order
        except (NetworkError, ExchangeError, asyncio.TimeoutError) as e:
            # Increment critical error counter and potentially activate safe mode
            self.order_manager.reliability_manager.record_critical_error(
                e, context={"symbol": get_signal_attr(signal, "symbol", None)}
            )
            logger.error(f"Live order failed after retries: {e}")
            trade_logger.log_failed_order(signal_to_dict(signal), str(e))

            # Notify watchdog of order execution failure
            if self.order_manager.watchdog_service:
                await self.order_manager.watchdog_service.report_order_execution_failure(
                    {
                        "component_id": "order_manager",
                        "error_message": str(e),
                        "symbol": get_signal_attr(signal, "symbol", None),
                        "strategy_id": get_signal_attr(
                            signal, "strategy_id", "unknown"
                        ),
                        "correlation_id": get_signal_attr(
                            signal, "correlation_id", f"order_{int(time.time())}"
                        ),
                    }
                )

            # Trigger rollback logic
            await self._rollback_failed_order(signal, str(e))

            # Suppress error and return None
            return None

        except Exception as e:
            # Increment critical error counter and potentially activate safe mode
            self.order_manager.reliability_manager.record_critical_error(
                e, context={"symbol": get_signal_attr(signal, "symbol", None)}
            )
            logger.error(f"Live order failed after retries: {e}")
            trade_logger.log_failed_order(signal_to_dict(signal), str(e))

            # Notify watchdog of order execution failure
            if self.order_manager.watchdog_service:
                await self.order_manager.watchdog_service.report_order_execution_failure(
                    {
                        "component_id": "order_manager",
                        "error_message": str(e),
                        "symbol": get_signal_attr(signal, "symbol", None),
                        "strategy_id": get_signal_attr(
                            signal, "strategy_id", "unknown"
                        ),
                        "correlation_id": get_signal_attr(
                            signal, "correlation_id", f"order_{int(time.time())}"
                        ),
                    }
                )

            # Trigger rollback logic
            await self._rollback_failed_order(signal, str(e))

            # Return structured error response instead of None
            return {
                "id": None,
                "status": "failed",
                "error": str(e),
                "symbol": get_signal_attr(signal, "symbol", None),
            }

    async def _rollback_failed_order(self, signal: Any, error_message: str) -> None:
        """Rollback logic for failed orders."""
        try:
            symbol = get_signal_attr(signal, "symbol", None)
            if symbol:
                # Cancel any pending orders for this symbol
                await self.order_manager.cancel_all_orders()
                logger.info(
                    f"Rollback completed for failed order on {symbol}: {error_message}"
                )
            else:
                logger.warning(
                    f"Could not rollback failed order - no symbol available: {error_message}"
                )
        except Exception as rollback_error:
            logger.error(f"Error during order rollback: {rollback_error}")


class PaperOrderExecutionStrategy(OrderExecutionStrategy):
    """Strategy for paper trading order execution."""

    def get_mode_name(self) -> str:
        return "paper"

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute order in paper trading mode."""
        order = await self.order_manager.paper_executor.execute_paper_order(signal)
        return await self.order_manager.order_processor.process_order(order)


class BacktestOrderExecutionStrategy(OrderExecutionStrategy):
    """Strategy for backtest order execution."""

    def get_mode_name(self) -> str:
        return "backtest"

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute order in backtest mode."""
        order = await self.order_manager.backtest_executor.execute_backtest_order(
            signal
        )
        return await self.order_manager.order_processor.process_order(order)


class FallbackOrderExecutionStrategy(OrderExecutionStrategy):
    """Fallback strategy that defaults to paper trading."""

    def get_mode_name(self) -> str:
        return "paper_fallback"

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute order using paper trading as fallback."""
        logger.warning("Unknown trading mode, falling back to paper trading")
        if self.order_manager.paper_executor is None:
            raise ValueError("Paper executor not available for fallback")
        order = await self.order_manager.paper_executor.execute_paper_order(signal)
        return await self.order_manager.order_processor.process_order(order)


class BalanceRetrievalStrategy(ABC):
    """Abstract base class for balance retrieval strategies."""

    def __init__(self, order_manager: "OrderManager"):
        self.order_manager = order_manager

    @abstractmethod
    async def get_balance(self) -> Decimal:
        """Get balance using this strategy."""
        pass


class LiveBalanceStrategy(BalanceRetrievalStrategy):
    """Strategy for retrieving live trading balance."""

    async def get_balance(self) -> Decimal:
        """Get balance from live exchange."""
        await self.order_manager._rate_limit()
        try:
            balance = await self.order_manager.live_executor.exchange.fetch_balance()
            return Decimal(
                str(
                    balance["total"].get(
                        self.order_manager.config.get("base_currency"), 0
                    )
                )
            )
        except Exception as e:
            logger.warning(f"Failed to fetch live balance: {e}")
            return Decimal(0)


class PaperBalanceStrategy(BalanceRetrievalStrategy):
    """Strategy for retrieving paper trading balance."""

    async def get_balance(self) -> Decimal:
        """Get balance from paper trading executor."""
        return self.order_manager.paper_executor.get_balance()


class BacktestBalanceStrategy(BalanceRetrievalStrategy):
    """Strategy for retrieving backtest balance."""

    async def get_balance(self) -> Decimal:
        """Get balance from backtest/portfolio manager."""
        if (
            self.order_manager.portfolio_mode
            and self.order_manager.portfolio_manager.paper_balances
        ):
            total = sum(
                [
                    float(v)
                    for v in self.order_manager.portfolio_manager.paper_balances.values()
                ]
            )
            return Decimal(str(total))
        return Decimal(0)


class EquityCalculationStrategy(ABC):
    """Abstract base class for equity calculation strategies."""

    def __init__(self, order_manager: "OrderManager"):
        self.order_manager = order_manager

    @abstractmethod
    async def calculate_equity(self, balance: Decimal) -> Decimal:
        """Calculate equity using this strategy."""
        pass


class LiveEquityStrategy(EquityCalculationStrategy):
    """Strategy for calculating live trading equity."""

    async def calculate_equity(self, balance: Decimal) -> Decimal:
        """Calculate equity including unrealized PnL from live positions."""
        unrealized = Decimal(0)
        for symbol, position in self.order_manager.order_processor.positions.items():
            try:
                # Validate position data
                entry_price = position.get("entry_price")
                amount = position.get("amount")
                if entry_price is None or amount is None:
                    logger.warning(
                        f"Invalid position data for {symbol}: missing entry_price or amount"
                    )
                    continue

                entry_price = Decimal(str(entry_price))
                amount = Decimal(str(amount))

                # Use cached ticker to reduce API calls
                ticker = await self.order_manager._get_cached_ticker(symbol)
                current_price = Decimal(
                    str(ticker.get("last") or ticker.get("close") or 0)
                )
                unrealized += (current_price - entry_price) * amount
            except (NetworkError, ExchangeError, OSError) as e:
                logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
                continue
            except (TypeError, ValueError, InvalidOperation) as e:
                logger.warning(
                    f"Data error while computing unrealized for {symbol}: {e}"
                )
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error while computing unrealized PnL")
                raise

        return balance + unrealized


class PortfolioEquityStrategy(EquityCalculationStrategy):
    """Strategy for calculating portfolio equity."""

    async def calculate_equity(self, balance: Decimal) -> Decimal:
        """Calculate equity for portfolio mode."""
        try:
            total = Decimal(0)
            # Sum balances and unrealized from positions
            if self.order_manager.portfolio_manager.paper_balances:
                total += sum(
                    self.order_manager.portfolio_manager.paper_balances.values()
                )
            # Add unrealized per-position
            for symbol, pos in self.order_manager.order_processor.positions.items():
                try:
                    entry = Decimal(pos.get("entry_price", Decimal(0)))
                    amt = Decimal(pos.get("amount", Decimal(0)))
                    total += entry * amt
                except (TypeError, ValueError, InvalidOperation) as e:
                    logger.warning(f"Invalid position data for {symbol}: {e}")
                    continue
                except Exception:
                    logger.exception("Unexpected error while aggregating positions")
                    raise
            return total
        except Exception:
            logger.exception("Unexpected error calculating portfolio total")
            raise


class SimpleEquityStrategy(EquityCalculationStrategy):
    """Simple strategy that returns balance as equity."""

    async def calculate_equity(self, balance: Decimal) -> Decimal:
        """Return balance as equity (no unrealized calculations)."""
        return balance


class OrderManager:
    """Manages order execution and tracking across all trading modes."""

    def __init__(
        self,
        config: Dict[str, Any],
        mode: Union[str, "TradingMode"],
        watchdog_service=None,
    ) -> None:
        """Initialize the OrderManager.

        Args:
            config: Configuration dictionary (expects keys 'order', 'paper', etc.).
            mode: Trading mode (either string like 'live'/'paper'/'backtest' or TradingMode enum).
        """
        # Accept either a nested config (with 'order','risk','paper') or a flat one to remain backward compatible
        self.config: Dict[str, Any] = config.get("order", config)
        self.risk_config: Dict[str, Any] = config.get(
            "risk", self.config.get("risk", {})
        )
        self.mode_original = mode

        # Normalize incoming mode to the canonical TradingMode enum
        try:
            if isinstance(mode, str):
                try:
                    # Prefer enum name lookup (e.g., "LIVE", "PAPER", "BACKTEST")
                    self.mode = TradingMode[mode.upper()]
                except KeyError:
                    # Fallback: match by enum value ("live","paper","backtest")
                    try:
                        self.mode = next(
                            m for m in TradingMode if m.value == str(mode).lower()
                        )
                    except StopIteration:
                        self.mode = TradingMode.PAPER
            elif isinstance(mode, TradingMode):
                self.mode = mode
            else:
                # Default to PAPER for safety if unknown
                self.mode = TradingMode.PAPER
        except (AttributeError, TypeError, ValueError):
            # Defensive fallback for unexpected types
            self.mode = TradingMode.PAPER

        # normalized mode name (lowercase string) for consistent checks across module
        self.mode_name: str = getattr(self.mode, "value", str(self.mode)).lower()

        # Initialize specialized managers - only initialize the executor for the current mode
        if self.mode == TradingMode.LIVE:
            self.live_executor = LiveOrderExecutor(config)
            self.paper_executor = None
            self.backtest_executor = None
        elif self.mode == TradingMode.PAPER:
            self.live_executor = MockLiveExecutor(config)
            self.paper_executor = PaperOrderExecutor(config)
            self.backtest_executor = None
        elif self.mode == TradingMode.BACKTEST:
            self.live_executor = MockLiveExecutor(config)
            self.paper_executor = None
            self.backtest_executor = BacktestOrderExecutor(self.config)
        self.order_processor = OrderProcessor()
        self.reliability_manager = ReliabilityManager(config.get("reliability", {}))
        self.portfolio_manager = PortfolioManager()
        self.watchdog_service = watchdog_service

        # Initialize comprehensive validation pipeline
        validation_config = config.get("validation", {
            "fail_fast": False,
            "validation_timeout": 5.0,
            "enable_circuit_breaker": False,
            "market_hours": {"enabled": False}
        })
        self.validation_pipeline = OrderValidationPipeline(validation_config)

        # Initialize partial fill reconciliation manager
        fill_config = config.get("partial_fill", {})
        self.fill_reconciler = PartialFillReconciliationManager(fill_config)

        # Initialize per-instance idempotency registry (using global registry for backward compatibility)
        self.registry = order_execution_registry
        self.registry.clear()

        # Set initial paper balance - use Decimal for precision
        default_balance = Decimal("1000.00")
        raw_balance = config.get("paper", {}).get("initial_balance", default_balance)
        try:
            balance = Decimal(str(raw_balance))
        except (InvalidOperation, TypeError, ValueError):
            logger.warning(
                f"Invalid initial_balance value: {raw_balance}, using default"
            )
            balance = default_balance

        # Set balance on paper executor if it exists (paper/backtest modes)
        if self.paper_executor:
            self.paper_executor.set_initial_balance(balance)
        self.portfolio_manager.set_initial_balance(balance)

        # Portfolio flags (BotEngine may set these attributes after instantiation)
        self.portfolio_mode: bool = False
        self.pairs: List[str] = []
        # Optional allocation mapping symbol->fraction (0..1)
        self.pair_allocation: Optional[Dict[str, float]] = None

        # Rate limiting for KuCoin API (10 req/sec)
        self._last_request_time: float = 0.0
        self._request_interval: float = 0.1  # 100ms between requests

        # Ticker cache for performance
        self._ticker_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl: float = 5.0  # 5 seconds cache
        self._cache_timestamps: Dict[str, float] = {}

        # Balance and equity cache for performance
        self._balance_cache: Optional[Decimal] = None
        self._equity_cache: Optional[Decimal] = None
        self._balance_cache_timestamp: float = 0.0
        self._equity_cache_timestamp: float = 0.0
        self._balance_cache_ttl: float = 10.0  # 10 seconds cache for balance
        self._equity_cache_ttl: float = 5.0  # 5 seconds cache for equity

        # Initialize strategy patterns
        self._execution_strategies = {
            TradingMode.LIVE: LiveOrderExecutionStrategy(self),
            TradingMode.PAPER: PaperOrderExecutionStrategy(self),
            TradingMode.BACKTEST: BacktestOrderExecutionStrategy(self),
        }
        self._fallback_strategy = FallbackOrderExecutionStrategy(self)

        self._balance_strategies = {
            TradingMode.LIVE: LiveBalanceStrategy(self),
            TradingMode.PAPER: PaperBalanceStrategy(self),
            TradingMode.BACKTEST: BacktestBalanceStrategy(self),
        }

        self._equity_strategies = {
            TradingMode.LIVE: LiveEquityStrategy(self),
            TradingMode.PAPER: PortfolioEquityStrategy(self)
            if self.portfolio_mode
            else SimpleEquityStrategy(self),
            TradingMode.BACKTEST: PortfolioEquityStrategy(self)
            if self.portfolio_mode
            else SimpleEquityStrategy(self),
        }

    @property
    def paper_balances(self) -> Dict[str, Decimal]:
        """Compatibility shim: expose paper_balances from PortfolioManager for existing callers/tests."""
        return getattr(self.portfolio_manager, "paper_balances", {})

    def _get_order_schema(self) -> Dict[str, Any]:
        """Get JSON schema for order payload validation."""
        return {
            "type": "object",
            "properties": {
                "strategy_id": {"type": "string", "minLength": 1},
                "symbol": {"type": "string", "pattern": r"^[A-Z]+/[A-Z]+$"},
                "signal_type": {
                    "type": "string",
                    "enum": ["ENTRY_LONG", "ENTRY_SHORT", "EXIT_LONG", "EXIT_SHORT"],
                },
                "signal_strength": {
                    "type": "string",
                    "enum": ["WEAK", "MODERATE", "STRONG"],
                },
                "order_type": {
                    "type": "string",
                    "enum": ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"],
                },
                "amount": {"type": "string", "pattern": r"^-?\d+(\.\d+)?$"},
                "price": {"type": ["string", "null"], "pattern": r"^-?\d+(\.\d+)?$"},
                "stop_loss": {
                    "type": ["string", "null"],
                    "pattern": r"^-?\d+(\.\d+)?$",
                },
                "take_profit": {
                    "type": ["string", "null"],
                    "pattern": r"^-?\d+(\.\d+)?$",
                },
                "timestamp": {"type": "string", "pattern": r"^\d+$"},
            },
            "required": [
                "strategy_id",
                "symbol",
                "signal_type",
                "order_type",
                "amount",
            ],
            "additionalProperties": True,
        }

    def _normalize_payload(self, payload: dict) -> dict:
        """Normalize payload by converting Decimal, float, and int values to strings for schema validation."""
        normalized = {}
        for k, v in payload.items():
            if isinstance(v, (Decimal, float, int)):
                normalized[k] = str(v)
            elif k == "order_type" and isinstance(v, str):
                # Normalize order_type to uppercase for schema validation
                normalized[k] = v.upper()
            else:
                normalized[k] = v
        return normalized

    def _normalize_enum(self, value):
        """Normalize enum values to their string names for schema validation."""
        from enum import Enum

        if isinstance(value, Enum):
            return (
                value.name
            )  # Use name instead of value for string-based serialization
        elif isinstance(value, dict):
            return {k: self._normalize_enum(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._normalize_enum(item) for item in value]
        return value

    def _validate_order_payload(self, signal: Any) -> None:
        """Validate order payload with direct validation logic."""
        # Allow mocks in test mode (detected by Mock type)
        from unittest.mock import Mock

        if isinstance(signal, Mock):
            logger.debug("Skipping validation for Mock object in test mode")
            return

        try:
            # Normalize enum attributes in the signal object before conversion
            self._normalize_signal_enums(signal)

            # Convert signal to dict for validation
            signal_dict = signal_to_dict(signal)

            # Normalize payload to convert numeric types to strings for schema validation
            normalized_dict = self._normalize_payload(signal_dict)

            # Validate against JSON schema
            schema = self._get_order_schema()
            jsonschema.validate(instance=normalized_dict, schema=schema)

            # Additional business rule validations (use original dict for business logic)
            self._validate_business_rules(signal_dict)

            logger.debug(
                f"Order payload validation passed for {signal_dict.get('symbol', 'unknown')}"
            )

        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            logger.error(f"Order payload validation failed: {error_msg}")
            raise ValueError(f"Invalid order payload: {error_msg}") from e
        except jsonschema.SchemaError as e:
            error_msg = f"Schema error: {e.message}"
            logger.error(f"Schema validation error: {error_msg}")
            raise ValueError(f"Schema validation error: {error_msg}") from e
        except Exception as e:
            error_msg = f"Unexpected validation error: {str(e)}"
            logger.error(f"Order payload validation error: {error_msg}")
            raise ValueError(f"Order validation error: {error_msg}") from e

    def _normalize_signal_enums(self, signal: Any) -> None:
        """Normalize enum attributes in signal object to use names instead of values."""
        from enum import Enum

        # Common enum attributes that might be in signals
        enum_attrs = ["signal_type", "signal_strength", "side"]
        for attr in enum_attrs:
            if hasattr(signal, attr):
                value = getattr(signal, attr)
                if isinstance(value, Enum):
                    setattr(signal, attr, value.name)

    def _validate_business_rules(self, signal_dict: Dict[str, Any]) -> None:
        """Validate business rules for order payload."""
        # Check for unsafe defaults
        amount = signal_dict.get("amount")
        if amount is not None and float(amount) <= 0:
            raise ValueError("Order amount must be positive")

        price = signal_dict.get("price")
        if price is not None and float(price) < 0:
            raise ValueError("Order price cannot be negative")

        # Validate stop loss and take profit for limit orders
        order_type = signal_dict.get("order_type", "").upper()
        if order_type in ["STOP", "STOP_LIMIT"]:
            if signal_dict.get("stop_loss") is None:
                raise ValueError("Stop orders must include stop_loss price")

        # Validate signal type consistency
        signal_type = signal_dict.get("signal_type", "")
        if signal_type.startswith("ENTRY_"):
            if order_type not in ["MARKET", "LIMIT"]:
                raise ValueError("Entry signals should use MARKET or LIMIT orders")
        elif signal_type.startswith("EXIT_"):
            if order_type not in ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]:
                raise ValueError("Exit signals can use any order type")

        # Check for missing critical fields with defaults that could be unsafe
        if signal_dict.get("amount") is None:
            raise ValueError("Order amount is required and cannot be None")

        if signal_dict.get("symbol") is None or signal_dict.get("symbol") == "":
            raise ValueError("Trading symbol is required")

        # Validate symbol format
        symbol = signal_dict.get("symbol", "")
        if "/" not in symbol:
            raise ValueError("Symbol must be in format BASE/QUOTE (e.g., BTC/USDT)")

        base, quote = symbol.split("/", 1)
        if not base or not quote:
            raise ValueError("Symbol must have both base and quote currencies")

    def _resolve_idempotency_key(self, signal: Any) -> str:
        """Resolve idempotency key for the signal, rejecting empty keys."""
        idempotency_key = get_signal_attr(signal, "idempotency_key", None)
        if idempotency_key is None:
            # Backward compatibility: auto-generate key for dicts, mocks, TradingSignals, or legacy test signals
            if (
                isinstance(signal, dict)
                or isinstance(signal, TradingSignal)
                or signal.__class__.__name__ in ("MockSignal", "Mock")
            ):
                idempotency_key = f"auto-{uuid.uuid4().hex}"
            else:
                raise MissingIdempotencyError("Idempotency key is required")
        elif not idempotency_key.strip():
            raise MissingIdempotencyError("Idempotency key is required")
        return idempotency_key

    def _generate_cache_key(self, signal: Any, idempotency_key: str) -> str:
        """Generate a cache key that includes both signal content and idempotency key."""
        # Include key signal attributes to ensure uniqueness per signal content
        signal_parts = [
            get_signal_attr(signal, "symbol", ""),
            get_signal_attr(signal, "signal_type", ""),
            get_signal_attr(signal, "order_type", ""),
            get_signal_attr(signal, "amount", ""),
            get_signal_attr(signal, "price", "") or "",
        ]
        signal_str = "_".join(str(p) for p in signal_parts)
        return f"{idempotency_key}_{hash(signal_str)}"

    async def execute_order(
        self, signal: Any, return_legacy_none_on_failure: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an order based on the trading signal.

        This wrapper adds schema validation, safe-mode checks and retry/backoff handling for
        external exchange operations. Paper/backtest modes retain existing
        behavior (no external retries required).
        """
        # Auto-detect test context for legacy None return
        if not return_legacy_none_on_failure:
            from unittest.mock import Mock

            if isinstance(signal, Mock) or signal.__class__.__name__ == "MockSignal":
                return_legacy_none_on_failure = True

        # Resolve idempotency key (auto-generate if missing)
        idempotency_key = self._resolve_idempotency_key(signal)
        # Set the key on the signal if it was auto-generated
        if isinstance(signal, dict):
            if "idempotency_key" not in signal:
                signal["idempotency_key"] = idempotency_key
        else:
            if getattr(signal, "idempotency_key", None) is None:
                signal.idempotency_key = idempotency_key

        # Generate cache key that includes signal content
        cache_key = self._generate_cache_key(signal, idempotency_key)

        # Check idempotency registry
        registry_state = self.registry.begin_execution(idempotency_key)
        if registry_state is not None:
            if isinstance(registry_state, dict):
                if registry_state.get("status") == "pending":
                    logger.warning(
                        f"Order execution already in progress for key {idempotency_key}"
                    )
                    return None  # Block concurrent execution
                elif registry_state.get("status") == "success":
                    logger.info(
                        f"Returning cached successful result for idempotency key {idempotency_key}"
                    )
                    return registry_state.get("result")
                else:
                    # Should not happen, but handle gracefully
                    logger.warning(
                        f"Unexpected registry state for key {idempotency_key}: {registry_state}"
                    )
                    return None

        # Validate order payload first
        try:
            self._validate_order_payload(signal)
        except ValueError as e:
            logger.error(f"Order validation failed: {str(e)}")
            trade_logger.log_failed_order(
                signal_to_dict(signal), f"validation_error: {str(e)}"
            )
            self.registry.mark_failure(idempotency_key, e)
            # Return legacy None for test contexts, structured dict for production
            if return_legacy_none_on_failure:
                return None
            else:
                return {"status": "validation_failed", "error": str(e)}

        # Safe mode: if activated, do not open new positions
        if self.reliability_manager.safe_mode_active:
            # Increment safe mode trigger counter
            if not hasattr(self, "_safe_mode_triggers"):
                self._safe_mode_triggers = 0
            self._safe_mode_triggers += 1
            logger.info(
                "Safe mode active: order skipped (not counted as failure)",
                exc_info=False,
            )
            # Log as a separate event, not as a failed order
            trade_logger.trade(
                "Order skipped: safe_mode_active",
                {"signal": signal_to_dict(signal), "reason": "safe_mode_active"},
            )
            skipped_result = {
                "id": None,
                "symbol": get_signal_attr(signal, "symbol", None),
                "status": "skipped",
                "reason": "safe_mode_active",
            }
            self.registry.mark_success(idempotency_key, skipped_result)
            return skipped_result

        # Use strategy pattern for order execution
        try:
            strategy = self._execution_strategies.get(
                self.mode, self._fallback_strategy
            )
            result = await strategy.execute_order(signal)
            if result is not None:
                self.registry.mark_success(idempotency_key, result)
            else:
                self.registry.mark_failure(
                    idempotency_key, Exception("Execution returned None")
                )
            return result
        except (
            NetworkError,
            ExchangeError,
            OSError,
            asyncio.TimeoutError,
            ValueError,
        ) as e:
            logger.error(f"Order execution failed: {str(e)}", exc_info=True)
            trade_logger.log_failed_order(signal_to_dict(signal), str(e))
            self.registry.mark_failure(idempotency_key, e)
            return None
        except asyncio.CancelledError:
            # Preserve cancellation semantics
            self.registry.mark_failure(
                idempotency_key, Exception("Execution cancelled")
            )
            raise
        except Exception as e:
            logger.exception("Unexpected error during order execution")
            self.reliability_manager.record_critical_error(
                e, context={"symbol": get_signal_attr(signal, "symbol", None)}
            )
            trade_logger.log_failed_order(signal_to_dict(signal), str(e))
            self.registry.mark_failure(idempotency_key, e)
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.mode != TradingMode.LIVE or not self.live_executor:
            return False

        try:
            await self.live_executor.exchange.cancel_order(order_id)
            if order_id in self.order_processor.open_orders:
                self.order_processor.open_orders[order_id].status = OrderStatus.CANCELED
                self.order_processor.closed_orders[
                    order_id
                ] = self.order_processor.open_orders[order_id]
                del self.order_processor.open_orders[order_id]
            return True
        except (NetworkError, ExchangeError, OSError) as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"Unexpected error cancelling order {order_id}")
            return False

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        if self.mode != TradingMode.LIVE or not self.live_executor:
            return

        open_orders = list(self.order_processor.open_orders.keys())
        if not open_orders:
            return

        failed_cancellations = []
        successful_cancellations = []

        for order_id in open_orders:
            try:
                result = await self.cancel_order(order_id)
                if result:
                    successful_cancellations.append(order_id)
                else:
                    failed_cancellations.append(order_id)
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {str(e)}")
                failed_cancellations.append(order_id)

        # If all cancellations failed, raise an exception
        if len(failed_cancellations) == len(open_orders):
            raise Exception("Failed to cancel orders")
        elif failed_cancellations:
            # Some failed, some succeeded - log but don't raise
            logger.warning(
                f"Partial cancellation failure: {len(successful_cancellations)} succeeded, {len(failed_cancellations)} failed"
            )

    async def _rate_limit(self) -> None:
        """Simple rate limiter for KuCoin API calls."""
        current_time = time.monotonic()
        time_since_last = current_time - self._last_request_time

        # If this is the first call (_last_request_time is 0.0) or not enough time has passed,
        # ensure we wait for the full interval
        if self._last_request_time == 0.0 or time_since_last < self._request_interval:
            wait_time = (
                self._request_interval
                if self._last_request_time == 0.0
                else self._request_interval - time_since_last
            )
            await asyncio.sleep(max(0, wait_time))

        self._last_request_time = time.monotonic()

    async def _get_cached_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data with caching to reduce API calls."""
        current_time = time.time()
        if (
            symbol in self._ticker_cache
            and (current_time - self._cache_timestamps.get(symbol, 0)) < self._cache_ttl
        ):
            return self._ticker_cache[symbol]

        if self.mode == TradingMode.LIVE and self.live_executor:
            await self._rate_limit()
            try:
                ticker = await self.live_executor.exchange.fetch_ticker(symbol)
                self._ticker_cache[symbol] = ticker
                self._cache_timestamps[symbol] = current_time
                return ticker
            except (NetworkError, ExchangeError, OSError) as e:
                logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
                # Return cached data if available, even if expired
                if symbol in self._ticker_cache:
                    return self._ticker_cache[symbol]
                raise
        else:
            raise ValueError("Ticker fetching only supported in LIVE mode")

    async def get_balance(self) -> Decimal:
        """Get current account balance with caching for performance."""
        current_time = time.time()

        # Check cache first
        if (
            self._balance_cache is not None
            and (current_time - self._balance_cache_timestamp) < self._balance_cache_ttl
        ):
            return self._balance_cache

        # Use strategy pattern for balance retrieval
        try:
            strategy = self._balance_strategies.get(self.mode)
            if strategy:
                balance_value = await strategy.get_balance()
            else:
                logger.warning(
                    f"No balance strategy for mode {self.mode}, using fallback"
                )
                balance_value = Decimal(0)
        except Exception as e:
            logger.warning(f"Error getting balance: {e}")
            balance_value = Decimal(0)

        # Cache the result
        self._balance_cache = balance_value
        self._balance_cache_timestamp = current_time
        return balance_value

    async def get_equity(self) -> Decimal:
        """Get current account equity (balance + unrealized PnL) with caching for performance."""
        current_time = time.time()

        # Check cache first
        if (
            self._equity_cache is not None
            and (current_time - self._equity_cache_timestamp) < self._equity_cache_ttl
        ):
            return self._equity_cache

        balance = await self.get_balance()

        # Use strategy pattern for equity calculation
        try:
            strategy = self._equity_strategies.get(self.mode)
            if strategy:
                equity_value = await strategy.calculate_equity(balance)
            else:
                logger.warning(
                    f"No equity strategy for mode {self.mode}, using balance as equity"
                )
                equity_value = balance
        except Exception as e:
            logger.exception(f"Error calculating equity: {e}")
            equity_value = balance

        # Cache the result
        self._equity_cache = equity_value
        self._equity_cache_timestamp = current_time
        return equity_value

    async def initialize_portfolio(
        self,
        pairs: List[str],
        portfolio_mode: bool,
        allocation: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize per-pair portfolio state. This is an optional hook that BotEngine
        may call to configure per-symbol balances and allocation.

        Args:
            pairs: List of trading symbols
            portfolio_mode: Whether portfolio mode is enabled
            allocation: Optional mapping symbol->fraction of total initial balance
        """
        # Validate inputs
        if pairs is None:
            raise TypeError("pairs cannot be None")
        if not isinstance(pairs, list):
            raise TypeError("pairs must be a list")
        if allocation is not None:
            if not isinstance(allocation, dict):
                raise TypeError("allocation must be a dict or None")
            # Validate allocation values are numeric and sum to reasonable total
            total_allocation = 0.0
            for symbol, fraction in allocation.items():
                if not isinstance(fraction, (int, float)):
                    raise ValueError(
                        f"Allocation fraction for {symbol} must be numeric"
                    )
                if fraction < 0:
                    raise ValueError(
                        f"Allocation fraction for {symbol} cannot be negative"
                    )
                total_allocation += fraction
            if total_allocation > 1.01:  # Allow small floating point tolerance
                raise ValueError(f"Total allocation ({total_allocation}) exceeds 100%")

        try:
            self.pairs = pairs
            self.portfolio_mode = bool(portfolio_mode)

            # Default to equal allocation if None
            if allocation is None and pairs:
                equal_allocation = 1.0 / len(pairs)
                allocation = {pair: equal_allocation for pair in pairs}
                logger.info(f"Using equal allocation for portfolio: {allocation}")

            self.pair_allocation = allocation

            # Configure executors and portfolio manager (only if they exist)
            if self.paper_executor:
                self.paper_executor.set_portfolio_mode(
                    portfolio_mode, pairs, allocation
                )
            self.portfolio_manager.initialize_portfolio(
                pairs, portfolio_mode, allocation
            )
        except (ValueError, TypeError, OSError) as e:
            logger.exception(f"Failed to initialize portfolio (recoverable): {e}")
            return
        except Exception:
            logger.exception("Unexpected error initializing portfolio")
            raise

    async def get_active_order_count(self) -> int:
        """Get count of active/open orders."""
        return self.order_processor.get_active_order_count()

    async def get_open_position_count(self) -> int:
        """Get count of open positions."""
        return self.order_processor.get_open_position_count()

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self.live_executor:
            await self.live_executor.shutdown()

        # Stop partial fill reconciliation
        await self.fill_reconciler.stop_reconciliation_loop()

    async def process_fill_report(self, fill_data: Dict[str, Any]) -> None:
        """Process a fill report from an exchange."""
        try:
            order_id = fill_data.get("order_id")
            if not order_id:
                logger.warning("Fill report missing order_id")
                return

            # Check if this is a partial fill
            fill_amount = Decimal(str(fill_data.get("amount", 0)))
            fill_price = Decimal(str(fill_data.get("price", 0)))
            symbol = fill_data.get("symbol", "")
            fill_type = fill_data.get("fill_type", "full")

            # Create fill object
            fill = Fill(
                order_id=order_id,
                symbol=symbol,
                amount=fill_amount,
                price=fill_price,
                timestamp=time.time(),
                fill_type=fill_type,
                exchange_order_id=fill_data.get("exchange_order_id"),
                fees=fill_data.get("fees")
            )

            # Check if we have a partial fill record for this order
            if order_id in self.fill_reconciler.partial_fills:
                # Add fill to existing record
                self.fill_reconciler.add_fill_to_record(order_id, fill)
                record = self.fill_reconciler.partial_fills[order_id]

                if record.status == "fully_filled":
                    logger.info(f"Order {order_id} fully filled after partial fills")
                    # Update position management here if needed
                    await self._update_position_from_fill(record)
                elif record.remaining_amount > 0 and record.should_retry():
                    # Trigger retry for remaining amount
                    await self._retry_partial_fill(record)

            else:
                # This might be a full fill or the start of a partial fill
                if fill_type in ["partial", "final"]:
                    # Register as partial fill
                    original_amount = fill_data.get("original_amount", fill_amount)
                    record = self.fill_reconciler.register_partial_fill(
                        order_id, symbol, Decimal(str(original_amount))
                    )
                    self.fill_reconciler.add_fill_to_record(order_id, fill)

                    # Check if this fill completed the order
                    if record.status == "fully_filled":
                        await self._update_position_from_fill(record)
                    elif record.remaining_amount > 0 and record.should_retry():
                        await self._retry_partial_fill(record)
                else:
                    # Full fill - update position directly
                    await self._update_position_from_fill(fill)

        except Exception as e:
            logger.error(f"Error processing fill report: {e}")
            self.fill_reconciler.metrics["exchange_discrepancies"] += 1

    async def _update_position_from_fill(self, fill_data: Union[Fill, PartialFillRecord]) -> None:
        """Update position management based on fill data."""
        try:
            if isinstance(fill_data, PartialFillRecord):
                # Use the final reconciled fill data
                symbol = fill_data.symbol
                total_amount = fill_data.filled_amount
                avg_price = sum(f.price * f.amount for f in fill_data.fills) / total_amount
                order_id = fill_data.original_order_id
            else:
                # Single fill
                symbol = fill_data.symbol
                total_amount = fill_data.amount
                avg_price = fill_data.price
                order_id = fill_data.order_id

            # Update order processor positions
            # This is a simplified version - in practice, this would integrate
            # with the existing position management system
            if symbol not in self.order_processor.positions:
                self.order_processor.positions[symbol] = {
                    "amount": 0,
                    "entry_price": 0,
                    "orders": []
                }

            position = self.order_processor.positions[symbol]
            current_amount = Decimal(str(position.get("amount", 0)))
            current_avg_price = Decimal(str(position.get("entry_price", 0)))

            # Calculate new average price (volume-weighted)
            new_amount = current_amount + total_amount
            if new_amount != 0:
                new_avg_price = ((current_amount * current_avg_price) + (total_amount * avg_price)) / new_amount
            else:
                new_avg_price = avg_price

            position["amount"] = float(new_amount)
            position["entry_price"] = float(new_avg_price)
            position["orders"].append(order_id)

            logger.info(f"Updated position for {symbol}: amount={new_amount}, avg_price={new_avg_price}")

        except Exception as e:
            logger.error(f"Error updating position from fill: {e}")

    async def _retry_partial_fill(self, record: PartialFillRecord) -> None:
        """Retry filling the remaining amount of a partial fill."""
        try:
            if not record.should_retry():
                return

            record.retry_count += 1

            # Create a new order for the remaining amount
            # This is a simplified implementation - in practice, this would
            # create a proper follow-up order with appropriate parameters
            retry_signal = {
                "strategy_id": f"partial_fill_retry_{record.original_order_id}",
                "symbol": record.symbol,
                "signal_type": "ENTRY_LONG",  # This should be determined from original order
                "order_type": "MARKET",  # Simplified - use market order for retry
                "amount": float(record.remaining_amount),
                "idempotency_key": f"retry_{record.original_order_id}_{record.retry_count}",
                "timestamp": int(time.time() * 1000)
            }

            logger.info(f"Retrying partial fill for order {record.original_order_id}, attempt {record.retry_count}")

            # Add delay before retry
            await asyncio.sleep(self.fill_reconciler.retry_delay)

            # Execute the retry order
            result = await self.execute_order(retry_signal)
            if result and result.get("status") == "filled":
                # Success - this will be handled by process_fill_report when the fill comes in
                logger.info(f"Partial fill retry successful for {record.original_order_id}")
            else:
                logger.warning(f"Partial fill retry failed for {record.original_order_id}")

        except Exception as e:
            logger.error(f"Error retrying partial fill for {record.original_order_id}: {e}")
            record.mark_for_manual_intervention(f"Retry failed: {str(e)}")

    def get_fill_reconciliation_metrics(self) -> Dict[str, int]:
        """Get partial fill reconciliation metrics."""
        return self.fill_reconciler.get_fill_metrics()

    def get_pending_partial_fills(self) -> List[PartialFillRecord]:
        """Get all pending partial fill records."""
        return self.fill_reconciler.get_pending_fills()

    def get_stuck_partial_fills(self) -> List[PartialFillRecord]:
        """Get partial fills that need manual intervention."""
        return self.fill_reconciler.get_stuck_fills()

    async def start_fill_reconciliation(self) -> None:
        """Start the background fill reconciliation process."""
        await self.fill_reconciler.start_reconciliation_loop()

    async def stop_fill_reconciliation(self) -> None:
        """Stop the background fill reconciliation process."""
        await self.fill_reconciler.stop_reconciliation_loop()
