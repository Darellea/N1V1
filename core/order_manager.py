"""
core/order_manager.py

Refactored OrderManager that orchestrates the new specialized classes.
Handles order execution, tracking, and management across all trading modes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING, Union
from decimal import Decimal, InvalidOperation
import random
import time
from utils.time import now_ms, to_ms
from typing import Callable
import os

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

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class OrderManager:
    """Manages order execution and tracking across all trading modes."""

    def __init__(self, config: Dict[str, Any], mode: Union[str, "TradingMode"]) -> None:
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
        self.live_executor = LiveOrderExecutor(self.config) if self.mode == TradingMode.LIVE else None
        self.paper_executor = PaperOrderExecutor(self.config)
        self.backtest_executor = BacktestOrderExecutor(self.config)
        self.order_processor = OrderProcessor()
        self.reliability_manager = ReliabilityManager(config.get("reliability", {}))
        self.portfolio_manager = PortfolioManager()

        # Set initial paper balance
        initial_balance = None
        paper_cfg = config.get("paper") if isinstance(config, dict) else None
        if isinstance(paper_cfg, dict):
            initial_balance = paper_cfg.get("initial_balance")
        else:
            cfg_paper = self.config.get("paper") if isinstance(self.config.get("paper", None), dict) else None
            if isinstance(cfg_paper, dict):
                initial_balance = cfg_paper.get("initial_balance")

        if initial_balance is None:
            # Last-resort: try top-level key (legacy)
            initial_balance = config.get("initial_balance", None) if isinstance(config, dict) else None
            if not initial_balance:
                initial_balance = self.config.get("initial_balance", None)

        self.paper_executor.set_initial_balance(initial_balance)
        self.portfolio_manager.set_initial_balance(initial_balance)

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

    @property
    def paper_balances(self) -> Dict[str, Decimal]:
        """Compatibility shim: expose paper_balances from PortfolioManager for existing callers/tests."""
        return getattr(self.portfolio_manager, "paper_balances", {})

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """
        Execute an order based on the trading signal.

        This wrapper adds safe-mode checks and retry/backoff handling for
        external exchange operations. Paper/backtest modes retain existing
        behavior (no external retries required).
        """
        # Safe mode: if activated, do not open new positions
        if self.reliability_manager.safe_mode_active:
            # Increment safe mode trigger counter
            if not hasattr(self, '_safe_mode_triggers'):
                self._safe_mode_triggers = 0
            self._safe_mode_triggers += 1
            logger.warning(f"Safe mode active: skipping new order execution (trigger #{self._safe_mode_triggers})", exc_info=False)
            trade_logger.log_failed_order(signal_to_dict(signal), "safe_mode_active")
            return {"id": None, "symbol": getattr(signal, "symbol", None), "status": "skipped", "reason": "safe_mode_active"}

        # Determine execution path
        try:
            if self.mode == TradingMode.BACKTEST:
                order = await self.backtest_executor.execute_backtest_order(signal)
                return await self.order_processor.process_order(order)
            elif self.mode == TradingMode.PAPER:
                order = await self.paper_executor.execute_paper_order(signal)
                return await self.order_processor.process_order(order)
            elif self.mode == TradingMode.LIVE:
                # For live mode, execute with retry/backoff for network-related errors.
                try:
                    order_response = await self.reliability_manager.retry_async(
                        lambda: self.live_executor.execute_live_order(signal),
                        exceptions=(NetworkError, ExchangeError, asyncio.TimeoutError, OSError),
                    )
                    order = self.order_processor.parse_order_response(order_response)
                    processed_order = await self.order_processor.process_order(order)
                    trade_logger.log_order(processed_order, self.mode_name)
                    return processed_order
                except Exception as e:
                    # Increment critical error counter and potentially activate safe mode
                    self.reliability_manager.record_critical_error(e, context={"symbol": getattr(signal, "symbol", None)})
                    logger.exception("Live order failed after retries")
                    trade_logger.log_failed_order(signal_to_dict(signal), str(e))
                    return None
            else:
                # Unknown mode: treat as paper for safety
                order = await self.paper_executor.execute_paper_order(signal)
                return await self.order_processor.process_order(order)
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
        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        # If this is the first call (_last_request_time is 0.0) or not enough time has passed,
        # ensure we wait for the full interval
        if self._last_request_time == 0.0 or time_since_last < self._request_interval:
            wait_time = self._request_interval if self._last_request_time == 0.0 else self._request_interval - time_since_last
            await asyncio.sleep(max(0, wait_time))

        self._last_request_time = time.time()

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

        # Check cache for live mode
        if (self.mode == TradingMode.LIVE and
            self._balance_cache is not None and
            (current_time - self._balance_cache_timestamp) < self._balance_cache_ttl):
            return self._balance_cache

        if self.mode == TradingMode.LIVE and self.live_executor:
            await self._rate_limit()
            try:
                balance = await self.live_executor.exchange.fetch_balance()
                balance_value = Decimal(str(balance["total"].get(self.config.get("base_currency"), 0)))
                # Cache the result
                self._balance_cache = balance_value
                self._balance_cache_timestamp = current_time
                return balance_value
            except Exception as e:
                logger.warning(f"Failed to fetch balance: {e}")
                return Decimal(0)
        elif self.mode == TradingMode.PAPER:
            balance_value = self.paper_executor.get_balance()
            # Cache paper balance as well
            self._balance_cache = balance_value
            self._balance_cache_timestamp = current_time
            return balance_value
        else:
            # Backtest doesn't track balance; aggregate closed PnL if requested elsewhere
            if self.portfolio_mode and self.portfolio_manager.paper_balances:
                total = sum([float(v) for v in self.portfolio_manager.paper_balances.values()])
                balance_value = Decimal(str(total))
            else:
                balance_value = Decimal(0)
            # Cache backtest balance
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

        if self.mode == TradingMode.LIVE and self.live_executor:
            # For live trading, calculate unrealized PnL from open positions using cached tickers
            unrealized = Decimal(0)
            for symbol, position in self.order_processor.positions.items():
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
                    ticker = await self._get_cached_ticker(symbol)
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
            equity_value = balance + unrealized
        else:
            # For paper/backtest aggregate per-pair unrealized
            if self.portfolio_mode:
                try:
                    total = Decimal(0)
                    # Sum balances and unrealized from positions (simple approach)
                    if self.portfolio_manager.paper_balances:
                        total += sum(self.portfolio_manager.paper_balances.values())
                    # Add unrealized per-position by using order.price as proxy (best-effort)
                    for symbol, pos in self.order_processor.positions.items():
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
                    equity_value = total
                except Exception:
                    # If aggregation fails unexpectedly, surface it
                    logger.exception("Unexpected error calculating portfolio total")
                    raise
            else:
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
