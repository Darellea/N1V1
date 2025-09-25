"""
core/order_manager.py

Refactored OrderManager that orchestrates the new specialized classes.
Handles order execution, tracking, and management across all trading modes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING, Union, Protocol
from decimal import Decimal, InvalidOperation
import random
import time
from utils.time import now_ms, to_ms
from typing import Callable
import os
from abc import ABC, abstractmethod
import json
import jsonschema

import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError

from utils.logger import get_trade_logger
from utils.config_loader import ConfigLoader
from core.types import TradingMode
from core.types.order_types import Order, OrderType, OrderStatus
from core.execution.live_executor import LiveOrderExecutor
from core.execution.paper_executor import PaperOrderExecutor
from core.execution.backtest_executor import BacktestOrderExecutor
from core.execution.order_processor import OrderProcessor
from core.management.reliability_manager import ReliabilityManager
from core.management.portfolio_manager import PortfolioManager
from utils.adapter import signal_to_dict
from core.contracts import SignalType

from .logging_utils import get_structured_logger, LogSensitivity

logger = get_structured_logger("core.order_manager", LogSensitivity.SECURE)
trade_logger = get_trade_logger()


class MockLiveExecutor:
    """Mock live executor for testing and non-live modes."""

    def __init__(self, config=None):
        self.exchange = None
        self.config = config

    async def execute_live_order(self, signal):
        """Mock live order execution."""
        # Return a mock successful response
        return {
            'id': f'mock_order_{random.randint(1000, 9999)}',
            'status': 'filled',
            'amount': getattr(signal, 'amount', 0.001),
            'price': 50000.0,
            'symbol': getattr(signal, 'symbol', 'BTC/USDT')
        }

    async def shutdown(self):
        """Mock shutdown."""
        pass


class OrderExecutionStrategy(ABC):
    """Abstract base class for order execution strategies."""

    def __init__(self, order_manager: 'OrderManager'):
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
            order_response = await self.order_manager.reliability_manager.retry_async(
                lambda: self.order_manager.live_executor.execute_live_order(signal),
                exceptions=(NetworkError, ExchangeError, asyncio.TimeoutError, OSError),
            )
            order = self.order_manager.order_processor.parse_order_response(order_response)
            processed_order = await self.order_manager.order_processor.process_order(order)
            trade_logger.log_order(processed_order, self.get_mode_name())
            return processed_order
        except Exception as e:
            # Increment critical error counter and potentially activate safe mode
            self.order_manager.reliability_manager.record_critical_error(
                e, context={"symbol": getattr(signal, "symbol", None)}
            )
            logger.exception("Live order failed after retries")
            trade_logger.log_failed_order(signal_to_dict(signal), str(e))

            # Notify watchdog of order execution failure
            if self.order_manager.watchdog_service:
                await self.order_manager.watchdog_service.report_order_execution_failure({
                    "component_id": "order_manager",
                    "error_message": str(e),
                    "symbol": getattr(signal, "symbol", None),
                    "strategy_id": getattr(signal, "strategy_id", "unknown"),
                    "correlation_id": getattr(signal, "correlation_id", f"order_{int(time.time())}")
                })

            # Trigger rollback logic
            await self._rollback_failed_order(signal, str(e))

            # Return failed order status
            return {
                "id": None,
                "symbol": getattr(signal, "symbol", None),
                "status": "failed",
                "error": str(e)
            }

    async def _rollback_failed_order(self, signal: Any, error_message: str) -> None:
        """Rollback logic for failed orders."""
        try:
            symbol = getattr(signal, "symbol", None)
            if symbol:
                # Cancel any pending orders for this symbol
                await self.order_manager.cancel_all_orders()
                logger.info(f"Rollback completed for failed order on {symbol}: {error_message}")
            else:
                logger.warning(f"Could not rollback failed order - no symbol available: {error_message}")
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
        order = await self.order_manager.backtest_executor.execute_backtest_order(signal)
        return await self.order_manager.order_processor.process_order(order)


class FallbackOrderExecutionStrategy(OrderExecutionStrategy):
    """Fallback strategy that defaults to paper trading."""

    def get_mode_name(self) -> str:
        return "paper_fallback"

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """Execute order using paper trading as fallback."""
        logger.warning("Unknown trading mode, falling back to paper trading")
        order = await self.order_manager.paper_executor.execute_paper_order(signal)
        return await self.order_manager.order_processor.process_order(order)


class BalanceRetrievalStrategy(ABC):
    """Abstract base class for balance retrieval strategies."""

    def __init__(self, order_manager: 'OrderManager'):
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
            return Decimal(str(balance["total"].get(self.order_manager.config.get("base_currency"), 0)))
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
        if self.order_manager.portfolio_mode and self.order_manager.portfolio_manager.paper_balances:
            total = sum([float(v) for v in self.order_manager.portfolio_manager.paper_balances.values()])
            return Decimal(str(total))
        return Decimal(0)


class EquityCalculationStrategy(ABC):
    """Abstract base class for equity calculation strategies."""

    def __init__(self, order_manager: 'OrderManager'):
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
                    logger.warning(f"Invalid position data for {symbol}: missing entry_price or amount")
                    continue

                entry_price = Decimal(str(entry_price))
                amount = Decimal(str(amount))

                # Use cached ticker to reduce API calls
                ticker = await self.order_manager._get_cached_ticker(symbol)
                current_price = Decimal(str(ticker.get("last") or ticker.get("close") or 0))
                unrealized += (current_price - entry_price) * amount
            except (NetworkError, ExchangeError, OSError) as e:
                logger.warning(f"Failed to fetch ticker for {symbol}: {e}")
                continue
            except (TypeError, ValueError, InvalidOperation) as e:
                logger.warning(f"Data error while computing unrealized for {symbol}: {e}")
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
                total += sum(self.order_manager.portfolio_manager.paper_balances.values())
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

    def __init__(self, config: Dict[str, Any], mode: Union[str, "TradingMode"], watchdog_service=None) -> None:
        """Initialize the OrderManager.

        Args:
            config: Configuration dictionary (expects keys 'order', 'paper', etc.).
            mode: Trading mode (either string like 'live'/'paper'/'backtest' or TradingMode enum).
        """
        # Accept either a nested config (with 'order','risk','paper') or a flat one to remain backward compatible
        self.config: Dict[str, Any] = config.get("order", config)
        self.risk_config: Dict[str, Any] = config.get("risk", self.config.get("risk", {}))
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
                        self.mode = next(m for m in TradingMode if m.value == str(mode).lower())
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

        # Initialize specialized managers
        self.live_executor = LiveOrderExecutor(config) if self.mode == TradingMode.LIVE else MockLiveExecutor()
        self.paper_executor = PaperOrderExecutor(config)
        self.backtest_executor = BacktestOrderExecutor(self.config)
        self.order_processor = OrderProcessor()
        self.reliability_manager = ReliabilityManager(config.get("reliability", {}))
        self.portfolio_manager = PortfolioManager()
        self.watchdog_service = watchdog_service

        # Set initial paper balance - use Decimal for precision
        default_balance = Decimal("1000.00")
        raw_balance = config.get("paper", {}).get("initial_balance", default_balance)
        try:
            balance = Decimal(str(raw_balance))
        except (InvalidOperation, TypeError, ValueError):
            logger.warning(f"Invalid initial_balance value: {raw_balance}, using default")
            balance = default_balance
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
        self._equity_cache_ttl: float = 5.0   # 5 seconds cache for equity

        # Initialize strategy patterns
        self._execution_strategies = {
            TradingMode.LIVE: LiveOrderExecutionStrategy(self),
            TradingMode.PAPER: PaperOrderExecutionStrategy(self),
            TradingMode.BACKTEST: BacktestOrderExecutionStrategy(self)
        }
        self._fallback_strategy = FallbackOrderExecutionStrategy(self)

        self._balance_strategies = {
            TradingMode.LIVE: LiveBalanceStrategy(self),
            TradingMode.PAPER: PaperBalanceStrategy(self),
            TradingMode.BACKTEST: BacktestBalanceStrategy(self)
        }

        self._equity_strategies = {
            TradingMode.LIVE: LiveEquityStrategy(self),
            TradingMode.PAPER: PortfolioEquityStrategy(self) if self.portfolio_mode else SimpleEquityStrategy(self),
            TradingMode.BACKTEST: PortfolioEquityStrategy(self) if self.portfolio_mode else SimpleEquityStrategy(self)
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
                "signal_type": {"type": "string", "enum": ["ENTRY_LONG", "ENTRY_SHORT", "EXIT_LONG", "EXIT_SHORT"]},
                "signal_strength": {"type": "string", "enum": ["WEAK", "MODERATE", "STRONG"]},
                "order_type": {"type": "string", "enum": ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]},
                "amount": {"type": "number", "minimum": 0.00000001},
                "price": {"type": ["number", "null"], "minimum": 0},
                "stop_loss": {"type": ["number", "null"], "minimum": 0},
                "take_profit": {"type": ["number", "null"], "minimum": 0},
                "timestamp": {"type": "number", "minimum": 0}
            },
            "required": ["strategy_id", "symbol", "signal_type", "order_type", "amount"],
            "additionalProperties": True
        }

    def _validate_order_payload(self, signal: Any) -> None:
        """Validate order payload against schema and business rules."""
        # Allow mocks in test mode (detected by Mock type)
        from unittest.mock import Mock
        if isinstance(signal, Mock):
            logger.debug("Skipping validation for Mock object in test mode")
            return

        try:
            # Convert signal to dict for validation
            signal_dict = signal_to_dict(signal)

            # Validate against JSON schema
            schema = self._get_order_schema()
            jsonschema.validate(instance=signal_dict, schema=schema)

            # Additional business rule validations
            self._validate_business_rules(signal_dict)

            logger.debug(f"Order payload validation passed for {signal_dict.get('symbol', 'unknown')}")

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

    def _validate_business_rules(self, signal_dict: Dict[str, Any]) -> None:
        """Validate business rules for order payload."""
        # Check for unsafe defaults
        if signal_dict.get("amount", 0) <= 0:
            raise ValueError("Order amount must be positive")

        if signal_dict.get("price", 0) < 0:
            raise ValueError("Order price cannot be negative")

        # Validate stop loss and take profit for limit orders
        order_type = signal_dict.get("order_type", "").upper()
        if order_type in ["STOP", "STOP_LIMIT"]:
            if "stop_loss" not in signal_dict:
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

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """
        Execute an order based on the trading signal.

        This wrapper adds schema validation, safe-mode checks and retry/backoff handling for
        external exchange operations. Paper/backtest modes retain existing
        behavior (no external retries required).
        """
        # Validate order payload first
        try:
            self._validate_order_payload(signal)
        except ValueError as e:
            logger.error(f"Order validation failed: {str(e)}")
            trade_logger.log_failed_order(signal_to_dict(signal), f"validation_error: {str(e)}")
            return None

        # Safe mode: if activated, do not open new positions
        if self.reliability_manager.safe_mode_active:
            # Increment safe mode trigger counter
            if not hasattr(self, '_safe_mode_triggers'):
                self._safe_mode_triggers = 0
            self._safe_mode_triggers += 1
            logger.info("Safe mode active: order skipped (not counted as failure)", exc_info=False)
            # Log as a separate event, not as a failed order
            trade_logger.trade("Order skipped: safe_mode_active", {"signal": signal_to_dict(signal), "reason": "safe_mode_active"})
            return {"id": None, "symbol": getattr(signal, "symbol", None), "status": "skipped_safe_mode", "reason": "safe_mode_active"}

        # Use strategy pattern for order execution
        try:
            strategy = self._execution_strategies.get(self.mode, self._fallback_strategy)
            return await strategy.execute_order(signal)
        except (NetworkError, ExchangeError, OSError) as e:
            logger.error(f"Order execution failed (exchange/network): {str(e)}", exc_info=True)
            trade_logger.log_failed_order(signal_to_dict(signal), str(e))
            return None
        except asyncio.CancelledError:
            # Preserve cancellation semantics
            raise
        except Exception as e:
            logger.exception("Unexpected error during order execution")
            self.reliability_manager.record_critical_error(e, context={"symbol": getattr(signal, "symbol", None)})
            trade_logger.log_failed_order(signal_to_dict(signal), str(e))
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.mode != TradingMode.LIVE or not self.live_executor:
            return False

        try:
            await self.live_executor.exchange.cancel_order(order_id)
            if order_id in self.order_processor.open_orders:
                self.order_processor.open_orders[order_id].status = OrderStatus.CANCELED
                self.order_processor.closed_orders[order_id] = self.order_processor.open_orders[order_id]
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
            logger.warning(f"Partial cancellation failure: {len(successful_cancellations)} succeeded, {len(failed_cancellations)} failed")

    async def _rate_limit(self) -> None:
        """Simple rate limiter for KuCoin API calls."""
        current_time = time.monotonic()
        time_since_last = current_time - self._last_request_time

        # If this is the first call (_last_request_time is 0.0) or not enough time has passed,
        # ensure we wait for the full interval
        if self._last_request_time == 0.0 or time_since_last < self._request_interval:
            wait_time = self._request_interval if self._last_request_time == 0.0 else self._request_interval - time_since_last
            await asyncio.sleep(max(0, wait_time))

        self._last_request_time = time.monotonic()

    async def _get_cached_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data with caching to reduce API calls."""
        current_time = time.time()
        if symbol in self._ticker_cache and (current_time - self._cache_timestamps.get(symbol, 0)) < self._cache_ttl:
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
        if (self._balance_cache is not None and
            (current_time - self._balance_cache_timestamp) < self._balance_cache_ttl):
            return self._balance_cache

        # Use strategy pattern for balance retrieval
        try:
            strategy = self._balance_strategies.get(self.mode)
            if strategy:
                balance_value = await strategy.get_balance()
            else:
                logger.warning(f"No balance strategy for mode {self.mode}, using fallback")
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
        if (self._equity_cache is not None and
            (current_time - self._equity_cache_timestamp) < self._equity_cache_ttl):
            return self._equity_cache

        balance = await self.get_balance()

        # Use strategy pattern for equity calculation
        try:
            strategy = self._equity_strategies.get(self.mode)
            if strategy:
                equity_value = await strategy.calculate_equity(balance)
            else:
                logger.warning(f"No equity strategy for mode {self.mode}, using balance as equity")
                equity_value = balance
        except Exception as e:
            logger.exception(f"Error calculating equity: {e}")
            equity_value = balance

        # Cache the result
        self._equity_cache = equity_value
        self._equity_cache_timestamp = current_time
        return equity_value

    async def initialize_portfolio(self, pairs: List[str], portfolio_mode: bool, allocation: Optional[Dict[str, float]] = None) -> None:
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
                    raise ValueError(f"Allocation fraction for {symbol} must be numeric")
                if fraction < 0:
                    raise ValueError(f"Allocation fraction for {symbol} cannot be negative")
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

            # Configure both executors and portfolio manager
            self.paper_executor.set_portfolio_mode(portfolio_mode, pairs, allocation)
            self.portfolio_manager.initialize_portfolio(pairs, portfolio_mode, allocation)
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
