"""
core/order_manager.py

Refactored OrderManager that orchestrates the new specialized classes.
Handles order execution, tracking, and management across all trading modes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING, Union
from decimal import Decimal
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
                except Exception:
                    # Fallback: match by enum value ("live","paper","backtest")
                    self.mode = next(m for m in TradingMode if m.value == str(mode).lower())
            elif isinstance(mode, TradingMode):
                self.mode = mode
            else:
                # Default to PAPER for safety if unknown
                self.mode = TradingMode.PAPER
        except Exception:
            # Defensive fallback
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
        try:
            initial_balance = config.get("paper", {}).get("initial_balance")
        except Exception:
            initial_balance = self.config.get("paper", {}).get("initial_balance", None) if isinstance(self.config.get("paper", None), dict) else None

        if initial_balance is None:
            # Last-resort: try top-level key (legacy)
            try:
                initial_balance = config.get("initial_balance", None) or self.config.get("initial_balance", None)
            except Exception:
                initial_balance = None

        self.paper_executor.set_initial_balance(initial_balance)
        self.portfolio_manager.set_initial_balance(initial_balance)

        # Portfolio flags (BotEngine may set these attributes after instantiation)
        self.portfolio_mode: bool = False
        self.pairs: List[str] = []
        # Optional allocation mapping symbol->fraction (0..1)
        self.pair_allocation: Optional[Dict[str, float]] = None

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
            logger.warning("Safe mode active: skipping new order execution", exc_info=False)
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
                        exceptions=(NetworkError, ExchangeError, TimeoutError, Exception),
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
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}", exc_info=True)
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
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        if self.mode != TradingMode.LIVE or not self.live_executor:
            return

        try:
            open_orders = list(self.order_processor.open_orders.keys())
            for order_id in open_orders:
                await self.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {str(e)}")

    async def get_balance(self) -> Decimal:
        """Get current account balance."""
        if self.mode == TradingMode.LIVE and self.live_executor:
            try:
                balance = await self.live_executor.exchange.fetch_balance()
                return Decimal(str(balance["total"].get(self.config.get("base_currency"), 0)))
            except Exception:
                return Decimal(0)
        elif self.mode == TradingMode.PAPER:
            return self.paper_executor.get_balance()
        else:
            # Backtest doesn't track balance; aggregate closed PnL if requested elsewhere
            if self.portfolio_mode and self.portfolio_manager.paper_balances:
                total = sum([float(v) for v in self.portfolio_manager.paper_balances.values()])
                return Decimal(str(total))
            return Decimal(0)

    async def get_equity(self) -> Decimal:
        """Get current account equity (balance + unrealized PnL)."""
        balance = await self.get_balance()

        if self.mode == TradingMode.LIVE and self.live_executor:
            # For live trading, we'd calculate unrealized PnL from open positions
            # This is a simplified version - real implementation would fetch current prices
            unrealized = Decimal(0)
            for symbol, position in self.order_processor.positions.items():
                try:
                    ticker = await self.live_executor.exchange.fetch_ticker(symbol)
                    current_price = Decimal(str(ticker.get("last") or ticker.get("close") or 0))
                    unrealized += (current_price - position["entry_price"]) * position["amount"]
                except Exception:
                    continue
            return balance + unrealized
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
                        except Exception:
                            continue
                    return total
                except Exception:
                    return balance
            return balance

    async def initialize_portfolio(self, pairs: List[str], portfolio_mode: bool, allocation: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize per-pair portfolio state. This is an optional hook that BotEngine
        may call to configure per-symbol balances and allocation.

        Args:
            pairs: List of trading symbols
            portfolio_mode: Whether portfolio mode is enabled
            allocation: Optional mapping symbol->fraction of total initial balance
        """
        try:
            self.pairs = pairs or []
            self.portfolio_mode = bool(portfolio_mode)
            self.pair_allocation = allocation or None

            # Configure both executors and portfolio manager
            self.paper_executor.set_portfolio_mode(portfolio_mode, pairs, allocation)
            self.portfolio_manager.initialize_portfolio(pairs, portfolio_mode, allocation)
        except Exception:
            logger.exception("Failed to initialize portfolio")
            return

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
