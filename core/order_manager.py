""" 
core/order_manager.py

Handles order execution, tracking, and management across all trading modes.
Implements live trading, paper trading, and backtesting order handling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING, Union
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
import random
import time
from typing import Callable

import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError

from utils.logger import TradeLogger
from utils.config_loader import ConfigLoader
from core.types import OrderType, OrderStatus
# Utility imported from risk manager for safe quantization when allocating per-pair balances
from risk.risk_manager import _safe_quantize

# TradingSignal is imported lazily in methods to avoid circular imports at module import time

logger = logging.getLogger(__name__)
trade_logger = TradeLogger()


@dataclass
class Order:
    """Dataclass representing an order."""

    id: str
    symbol: str
    type: OrderType
    side: str  # 'buy' or 'sell'
    amount: Decimal
    price: Optional[Decimal] = None
    status: OrderStatus = OrderStatus.OPEN
    filled: Decimal = Decimal(0)
    remaining: Decimal = Decimal(0)
    cost: Decimal = Decimal(0)
    fee: Dict = None
    trailing_stop: Optional[Decimal] = None
    timestamp: int = 0
    params: Dict = None


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
        self.mode: Union[str, "TradingMode"] = mode
        self.exchange: Optional[ccxt.Exchange] = None

        # Portfolio-mode support: per-symbol paper balances and overall paper_balance fallback
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

        try:
            self.paper_balance: Decimal = Decimal(initial_balance) if initial_balance is not None else Decimal("0")
        except Exception:
            self.paper_balance = Decimal("0")

        # Per-pair state containers (populated if portfolio_mode/pairs are configured)
        self.paper_balances: Dict[str, Decimal] = {}
        self.open_orders: Dict[str, Order] = {}
        self.closed_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_count: int = 0

        # Portfolio flags (BotEngine may set these attributes after instantiation)
        self.portfolio_mode: bool = False
        self.pairs: List[str] = []
        # Optional allocation mapping symbol->fraction (0..1)
        self.pair_allocation: Optional[Dict[str, float]] = None

        # Initialize exchange for live trading (handle enum/string)
        is_live = False
        try:
            # Avoid importing TradingMode at runtime; accept either enum or string
            is_live = getattr(self.mode, "name", str(self.mode)).lower() == "live"
        except Exception:
            is_live = str(self.mode).lower() == "live"

        if is_live:
            self._initialize_exchange()

        # Reliability defaults for retry/backoff and safe-mode.
        # Read overrides from provided config under top-level "reliability" key if present.
        self._reliability = {}
        try:
            cfg_rel = config.get("reliability", {}) if isinstance(config, dict) else {}
        except Exception:
            cfg_rel = {}
        self._reliability.update(
            {
                "max_retries": int(cfg_rel.get("max_retries", 3)),
                "backoff_base": float(cfg_rel.get("backoff_base", 0.5)),
                "max_backoff": float(cfg_rel.get("max_backoff", 10.0)),
                "safe_mode_threshold": int(cfg_rel.get("safe_mode_threshold", 5)),
                "close_positions_on_safe": bool(cfg_rel.get("close_positions_on_safe", False)),
            }
        )
        # Safe-mode state
        self.safe_mode_active = False
        self.critical_error_count = 0

    def _initialize_exchange(self) -> None:
        """Initialize the exchange connection."""
        exchange_config = {
            "apiKey": self.config["exchange"]["api_key"],
            "secret": self.config["exchange"]["api_secret"],
            "password": self.config["exchange"]["api_passphrase"],
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        }
        exchange_class = getattr(ccxt, self.config["exchange"]["name"])
        self.exchange = exchange_class(exchange_config)

    async def execute_order(self, signal: Any) -> Optional[Dict[str, Any]]:
        """
        Execute an order based on the trading signal.

        This wrapper adds safe-mode checks and retry/backoff handling for
        external exchange operations. Paper/backtest modes retain existing
        behavior (no external retries required).
        """
        # Safe mode: if activated, do not open new positions
        if getattr(self, "safe_mode_active", False):
            logger.warning("Safe mode active: skipping new order execution", exc_info=False)
            trade_logger.log_failed_order(signal, "safe_mode_active")
            return {"id": None, "symbol": getattr(signal, "symbol", None), "status": "skipped", "reason": "safe_mode_active"}

        # Determine execution path
        try:
            mode_name = getattr(self.mode, "name", str(self.mode)).lower()
            if mode_name == "backtest":
                return await self._execute_backtest_order(signal)
            elif mode_name == "paper":
                return await self._execute_paper_order(signal)
            else:  # live
                # For live mode, execute with retry/backoff for network-related errors.
                try:
                    return await self._retry_async(
                        lambda: self._execute_live_order(signal),
                        retries=self._reliability.get("max_retries", 3),
                        base_backoff=self._reliability.get("backoff_base", 0.5),
                        max_backoff=self._reliability.get("max_backoff", 10.0),
                        exceptions=(NetworkError, ExchangeError, TimeoutError, Exception),
                    )
                except Exception as e:
                    # Increment critical error counter and potentially activate safe mode
                    self._record_critical_error(e, context={"symbol": getattr(signal, "symbol", None)})
                    logger.exception("Live order failed after retries")
                    trade_logger.log_failed_order(signal, str(e))
                    return None
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}", exc_info=True)
            trade_logger.log_failed_order(signal, str(e))
            return None

    async def _retry_async(
        self,
        coro_factory: Callable[[], "Coroutine[Any, Any, Any]"],
        retries: int = 3,
        base_backoff: float = 0.5,
        max_backoff: float = 10.0,
        exceptions: tuple = (Exception,),
    ) -> Any:
        """
        Execute an async operation with exponential backoff and jitter.

        Args:
            coro_factory: A zero-arg callable returning the coroutine to execute.
            retries: Number of retry attempts (not counting the initial try).
            base_backoff: Base backoff in seconds (will be doubled each retry).
            max_backoff: Maximum backoff cap.
            exceptions: Tuple of exception types that should trigger a retry.

        Returns:
            Result of the coroutine if successful.

        Raises:
            The last exception if all retries are exhausted.
        """
        attempt = 0
        while True:
            try:
                attempt += 1
                coro = coro_factory()
                return await coro
            except exceptions as e:
                # If no retries left, re-raise
                if attempt > retries:
                    logger.error(f"Retry exhausted after {attempt-1} retries: {str(e)}", exc_info=True)
                    raise
                # Compute backoff with jitter
                backoff = min(max_backoff, base_backoff * (2 ** (attempt - 1)))
                jitter = backoff * 0.1
                sleep_for = backoff + random.uniform(-jitter, jitter)
                # Log retry attempt
                logger.warning(
                    f"Operation failed (attempt {attempt}/{retries}). Retrying in {sleep_for:.2f}s. Error: {str(e)}",
                    exc_info=False,
                )
                trade_logger.trade(
                    f"Retrying operation (attempt {attempt}/{retries})",
                    {"attempt": attempt, "error": str(e), "backoff": sleep_for},
                )
                await asyncio.sleep(max(0.0, sleep_for))
                continue

    def _record_critical_error(self, exc: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a critical error occurrence and activate safe mode if thresholds exceeded.

        Args:
            exc: Exception instance
            context: Optional contextual info (symbol, operation, etc.)
        """
        try:
            # Initialize counters if not present
            if not hasattr(self, "critical_error_count"):
                self.critical_error_count = 0
            if not hasattr(self, "_reliability"):
                # ensure defaults present
                self._init_reliability_defaults()

            self.critical_error_count += 1
            trade_logger.trade("Critical error recorded", {"count": self.critical_error_count, "error": str(exc), "context": context})
            logger.error(f"Critical error #{self.critical_error_count}: {str(exc)}", exc_info=False)

            # Activate safe mode if threshold exceeded
            threshold = int(self._reliability.get("safe_mode_threshold", 5))
            if self.critical_error_count >= threshold and not getattr(self, "safe_mode_active", False):
                self._activate_safe_mode(reason=str(exc), context=context)
        except Exception:
            logger.exception("Failed to record critical error")

    def _activate_safe_mode(self, reason: str = "", context: Optional[Dict[str, Any]] = None) -> None:
        """
        Enable safe mode which disables opening new positions and optionally closes existing ones.
        """
        try:
            self.safe_mode_active = True
            trade_logger.trade("Safe mode activated", {"reason": reason, "context": context})
            logger.critical(f"Safe mode activated due to: {reason}")

            # Optionally close existing positions based on configuration
            close_on_safe = bool(self._reliability.get("close_positions_on_safe", False))
            if close_on_safe:
                logger.info("Safe mode configured to close existing positions; attempting to cancel/close")
                # Attempt to cancel outstanding orders first
                try:
                    asyncio.create_task(self.cancel_all_orders())
                except Exception:
                    logger.exception("Failed to schedule cancel_all_orders")
                # TODO: implement graceful close of open positions (market sells) if desired.
        except Exception:
            logger.exception("Failed to activate safe mode")

    async def _execute_live_order(self, signal: Any) -> Dict[str, Any]:
        """
        Execute a live order on the exchange.

        Args:
            signal: Object providing order details (see execute_order docstring).

        Returns:
            Dictionary containing order execution details.
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized for live trading")

        order_params = {
            "symbol": getattr(
                signal,
                "symbol",
                signal.get("symbol") if isinstance(signal, dict) else None,
            ),
            "type": getattr(getattr(signal, "order_type", None), "value", None)
            if not isinstance(signal, dict)
            else signal.get("order_type"),
            "side": getattr(
                signal, "side", signal.get("side") if isinstance(signal, dict) else None
            ),
            "amount": float(
                getattr(
                    signal,
                    "amount",
                    signal.get("amount") if isinstance(signal, dict) else 0,
                )
            ),
            "price": float(
                getattr(
                    signal,
                    "price",
                    signal.get("price") if isinstance(signal, dict) else None,
                )
            )
            if getattr(
                signal,
                "price",
                signal.get("price", None) if isinstance(signal, dict) else None,
            )
            else None,
            "params": getattr(
                signal,
                "params",
                signal.get("params") if isinstance(signal, dict) else {},
            )
            or {},
        }

        try:
            # Execute the order
            response = await self.exchange.create_order(**order_params)
            order = self._parse_order_response(response)

            # Process the order
            processed_order = await self._process_order(order)
            trade_logger.log_order(processed_order, self.mode)
            return processed_order

        except (NetworkError, ExchangeError) as e:
            logger.error(f"Exchange error during order execution: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during live order execution: {str(e)}")
            raise

    async def _execute_paper_order(self, signal: Any) -> Dict[str, Any]:
        """
        Simulate order execution for paper trading.

        Args:
            signal: Object providing order details.

        Returns:
            Dictionary containing simulated order execution details.
        """
        # Calculate fees and slippage
        fee = self._calculate_fee(signal)
        executed_price = self._apply_slippage(signal)

        # Calculate order cost
        cost = Decimal(signal.amount) * Decimal(executed_price)
        total_cost = cost + fee if signal.side == "buy" else cost - fee

        # Check balance
        if signal.side == "buy" and total_cost > self.paper_balance:
            raise ValueError("Insufficient balance for paper trading order")

        # Create simulated order
        order = Order(
            id=f"paper_{self.trade_count}",
            symbol=signal.symbol,
            type=signal.order_type,
            side=signal.side,
            amount=Decimal(signal.amount),
            price=Decimal(executed_price),
            status=OrderStatus.FILLED,
            filled=Decimal(signal.amount),
            remaining=Decimal(0),
            cost=cost,
            params=getattr(signal, "params", {}) or ({"stop_loss": getattr(signal, "stop_loss", None)}),
            fee={"cost": float(fee), "currency": self.config["base_currency"]},
            trailing_stop=(
                Decimal(str(signal.trailing_stop.get("price")))
                if getattr(signal, "trailing_stop", None)
                and isinstance(signal.trailing_stop, dict)
                and signal.trailing_stop.get("price")
                else None
            ),
            timestamp=int(time.time() * 1000),
        )

        # Update paper balance (support per-pair balances when portfolio_mode enabled)
        symbol = getattr(signal, "symbol", None) or (signal.get("symbol") if isinstance(signal, dict) else None)
        if self.portfolio_mode and symbol:
            # ensure per-symbol balance exists
            bal = self.paper_balances.setdefault(symbol, Decimal(self.paper_balance))
            if signal.side == "buy":
                bal = bal - total_cost
            else:
                bal = bal + total_cost
            self.paper_balances[symbol] = _safe_quantize(bal)
        else:
            if signal.side == "buy":
                self.paper_balance -= total_cost
            else:
                self.paper_balance += total_cost

        # Process the order
        processed_order = await self._process_order(order)
        trade_logger.log_order(processed_order, self.mode)
        return processed_order

    async def _execute_backtest_order(self, signal: Any) -> Dict[str, Any]:
        """
        Simulate order execution for backtesting.

        Args:
            signal: Object providing order details.

        Returns:
            Dictionary containing backtest order execution details.
        """
        # Backtest orders are similar to paper trading but with historical data
        executed_price = signal.price  # For backtest, we use the exact price
        fee = self._calculate_fee(signal)

        order = Order(
            id=f"backtest_{self.trade_count}",
            symbol=signal.symbol,
            type=signal.order_type,
            side=signal.side,
            amount=Decimal(signal.amount),
            price=Decimal(executed_price),
            status=OrderStatus.FILLED,
            filled=Decimal(signal.amount),
            remaining=Decimal(0),
            cost=Decimal(signal.amount) * Decimal(executed_price),
            params=getattr(signal, "params", {}) or ({"stop_loss": getattr(signal, "stop_loss", None)}),
            fee={"cost": float(fee), "currency": self.config["base_currency"]},
            trailing_stop=(
                Decimal(str(signal.trailing_stop.get("price")))
                if getattr(signal, "trailing_stop", None)
                and isinstance(signal.trailing_stop, dict)
                and signal.trailing_stop.get("price")
                else None
            ),
            timestamp=signal.timestamp,
        )

        # Process the order (no balance tracking in backtest). Still record per-pair pnl if portfolio_mode.
        processed_order = await self._process_order(order)
        trade_logger.log_order(processed_order, self.mode)
        return processed_order

    def _parse_order_response(self, response: Dict) -> Order:
        """
        Parse exchange order response into our Order dataclass.

        Args:
            response: Raw exchange order response

        Returns:
            Parsed Order object
        """
        return Order(
            id=str(response["id"]),
            symbol=response["symbol"],
            type=OrderType(response["type"]),
            side=response["side"],
            amount=Decimal(str(response["amount"])),
            price=Decimal(str(response["price"])) if response["price"] else None,
            status=OrderStatus(response["status"]),
            filled=Decimal(str(response["filled"])),
            remaining=Decimal(str(response["remaining"])),
            cost=Decimal(str(response["cost"])) if response["cost"] else Decimal(0),
            fee=response.get("fee"),
            timestamp=response["timestamp"],
            params=response.get("params"),
        )

    async def _process_order(self, order: Order) -> Dict[str, Any]:
        """
        Process an order after execution (live or simulated).

        Args:
            order: The executed Order dataclass instance.

        Returns:
            Dictionary with processed order details including PnL.
        """
        self.trade_count += 1

        # Store the order
        if order.status == OrderStatus.FILLED:
            self.closed_orders[order.id] = order
            if order.id in self.open_orders:
                del self.open_orders[order.id]

            # Update position tracking
            self._update_positions(order)
        else:
            self.open_orders[order.id] = order

        # Calculate PnL if this was a closing trade
        pnl = self._calculate_pnl(order) if order.side == "sell" else None

        # Optionally compute dynamic take-profit when requested via order.params
        take_profit_val: Optional[float] = None
        try:
            params = order.params or {}
            dynamic_tp = params.get("dynamic_tp") or params.get("dynamic_take_profit")
            # prefer explicit stop_loss from params; fallback to trailing_stop on position
            stop_loss_param = params.get("stop_loss", None)
            if stop_loss_param is None:
                # get position's trailing stop if any (useful for backtest/paper)
                pos = self.positions.get(order.symbol, {})
                stop_loss_param = pos.get("trailing_stop", None)

            if dynamic_tp and order.price and stop_loss_param is not None:
                entry = Decimal(str(order.price))
                stop = Decimal(str(stop_loss_param))
                # Determine direction and risk
                if order.side == "buy":
                    risk = entry - stop
                    # trend_strength can be passed via params (0..1); otherwise try to estimate
                    try:
                        if params.get("trend_strength") is not None:
                            trend_strength = float(params.get("trend_strength") or 0.0)
                        else:
                            # attempt to estimate trend strength from market data (live mode)
                            trend_strength = await self._estimate_trend_strength(order.symbol)
                    except Exception:
                        trend_strength = 0.0
                    trend_strength = max(0.0, min(1.0, float(trend_strength)))
                    multiplier = float(self.risk_config.get("risk_reward_ratio", 2.0)) * (
                        1.0 + trend_strength
                    )
                    tp_dec = entry + Decimal(str(multiplier)) * Decimal(str(risk))
                    take_profit_val = float(_safe_quantize(tp_dec))
                else:
                    # short
                    risk = stop - entry
                    try:
                        if params.get("trend_strength") is not None:
                            trend_strength = float(params.get("trend_strength") or 0.0)
                        else:
                            # attempt to estimate trend strength (magnitude) for short
                            trend_strength = await self._estimate_trend_strength(order.symbol)
                    except Exception:
                        trend_strength = 0.0
                    trend_strength = max(0.0, min(1.0, float(trend_strength)))
                    multiplier = float(self.risk_config.get("risk_reward_ratio", 2.0)) * (
                        1.0 + trend_strength
                    )
                    tp_dec = entry - Decimal(str(multiplier)) * Decimal(str(risk))
                    take_profit_val = float(_safe_quantize(tp_dec))
        except Exception:
            # If dynamic TP fails, leave take_profit_val as None
            take_profit_val = None

        result = {
            "id": order.id,
            "symbol": order.symbol,
            "type": order.type.value,
            "side": order.side,
            "amount": float(order.amount),
            "price": float(order.price) if order.price else None,
            "status": order.status.value,
            "cost": float(order.cost),
            "fee": order.fee,
            "timestamp": order.timestamp,
            "pnl": pnl,
            "mode": self.mode,
        }
        if take_profit_val is not None:
            result["take_profit"] = take_profit_val

        return result

    def _update_positions(self, order: Order) -> None:
        """Update position tracking based on filled orders."""
        position = self.positions.get(
            order.symbol,
            {
                "amount": Decimal(0),
                "entry_price": Decimal(0),
                "entry_cost": Decimal(0),
                "trailing_stop": None,
            },
        )

        if order.side == "buy":
            new_amount = position["amount"] + order.filled
            new_cost = position["entry_cost"] + order.cost
            # Determine trailing stop value from order (if provided) or keep existing
            trailing = position.get("trailing_stop")
            if getattr(order, "trailing_stop", None) is not None:
                trailing = order.trailing_stop
            elif order.params and isinstance(order.params, dict):
                # support params-based trailing_stop price key
                ts_param = order.params.get("trailing_stop")
                if isinstance(ts_param, dict) and ts_param.get("price"):
                    trailing = Decimal(str(ts_param.get("price")))
            position.update(
                {
                    "amount": new_amount,
                    "entry_price": new_cost / new_amount
                    if new_amount > 0
                    else Decimal(0),
                    "entry_cost": new_cost,
                    "trailing_stop": trailing,
                }
            )
            self.positions[order.symbol] = position
        else:
            position["amount"] -= order.filled
            if position["amount"] <= 0:
                del self.positions[order.symbol]
            else:
                self.positions[order.symbol] = position

    def _calculate_pnl(self, order: Order) -> Optional[float]:
        """Calculate PnL for a sell order."""
        if order.side != "sell" or order.symbol not in self.positions:
            return None

        position = self.positions[order.symbol]
        entry_value = position["entry_price"] * order.filled
        exit_value = order.price * order.filled
        gross_pnl = exit_value - entry_value
        fee = Decimal(order.fee["cost"]) if order.fee else Decimal(0)
        net_pnl = gross_pnl - fee

        return float(net_pnl)

    def _calculate_fee(self, signal: Any) -> Decimal:
        """Calculate trading fee based on config.

        Args:
            signal: Object providing an 'amount' attribute or key.

        Returns:
            Decimal fee amount.
        """
        fee_rate = Decimal(self.config["trade_fee"])
        amt = getattr(
            signal, "amount", signal.get("amount") if isinstance(signal, dict) else 0
        )
        return Decimal(amt) * fee_rate

    def _apply_slippage(self, signal: Any) -> float:
        """Apply simulated slippage to order price.

        Args:
            signal: Object providing 'price' and 'side'.

        Returns:
            Adjusted price (float) including slippage.
        """
        slippage = Decimal(self.config["slippage"])
        price = getattr(
            signal, "price", signal.get("price") if isinstance(signal, dict) else None
        )
        side = getattr(
            signal, "side", signal.get("side") if isinstance(signal, dict) else None
        )
        if price is None:
            raise ValueError("Signal price required for slippage calculation")
        price_d = Decimal(price)
        if side == "buy":
            return float(price_d * (1 + slippage))
        else:
            return float(price_d * (1 - slippage))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.mode != "live":
            return False

        try:
            await self.exchange.cancel_order(order_id)
            if order_id in self.open_orders:
                self.open_orders[order_id].status = OrderStatus.CANCELED
                self.closed_orders[order_id] = self.open_orders[order_id]
                del self.open_orders[order_id]
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        if self.mode != "live":
            return

        try:
            open_orders = list(self.open_orders.keys())
            for order_id in open_orders:
                await self.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {str(e)}")

    async def get_balance(self) -> Decimal:
        """Get current account balance."""
        if self.mode == "live":
            balance = await self.exchange.fetch_balance()
            return Decimal(str(balance["total"][self.config["base_currency"]]))
        elif self.mode == "paper":
            if self.portfolio_mode and self.paper_balances:
                # Aggregate per-pair paper balances
                total = sum([float(v) for v in self.paper_balances.values()])
                return Decimal(str(total))
            return self.paper_balance
        else:
            # Backtest doesn't track balance; aggregate closed PnL if requested elsewhere
            if self.portfolio_mode and self.paper_balances:
                total = sum([float(v) for v in self.paper_balances.values()])
                return Decimal(str(total))
            return Decimal(0)

    async def get_equity(self) -> Decimal:
        """Get current account equity (balance + unrealized PnL)."""
        balance = await self.get_balance()

        if self.mode == "live":
            # For live trading, we'd calculate unrealized PnL from open positions
            # This is a simplified version - real implementation would fetch current prices
            unrealized = Decimal(0)
            for symbol, position in self.positions.items():
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = Decimal(str(ticker["last"]))
                unrealized += (current_price - position["entry_price"]) * position[
                    "amount"
                ]

            return balance + unrealized
        else:
            # For paper/backtest aggregate per-pair unrealized by using positions and latest prices if exchange not available.
            if self.portfolio_mode:
                try:
                    total = Decimal(0)
                    # Sum balances and unrealized from positions (simple approach)
                    if self.paper_balances:
                        total += sum(self.paper_balances.values())
                    # Add unrealized per-position by using order.price as proxy (best-effort)
                    for symbol, pos in self.positions.items():
                        try:
                            entry = Decimal(pos.get("entry_price", Decimal(0)))
                            amt = Decimal(pos.get("amount", Decimal(0)))
                            total += entry * amt
                        except Exception:
                            continue
                    return total
                except Exception:
                    return balance
            return balance  # Paper/backtest doesn't track unrealized PnL in this simplified version

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

            if not self.portfolio_mode or not self.pairs:
                return

            # Determine initial capital per pair. Use allocation if provided, else equal split.
            total = Decimal(str(self.paper_balance)) if self.paper_balance is not None else Decimal("0")
            if self.pair_allocation:
                for s in self.pairs:
                    frac = Decimal(str(self.pair_allocation.get(s, 0)))
                    self.paper_balances[s] = _safe_quantize(total * frac)
            else:
                per = _safe_quantize(total / Decimal(len(self.pairs)))
                for s in self.pairs:
                    self.paper_balances[s] = per
        except Exception:
            logger.exception("Failed to initialize portfolio")
            return

    async def _evaluate_trailing_stops_live(self) -> None:
        """
        Check trailing stops for open positions in live mode and trigger market sells
        if a trailing stop is hit. This attempts to reuse the existing live execution
        path by constructing a minimal signal dict.
        """
        # Only meaningful in live mode with an initialized exchange
        if not self.exchange or self.mode != "live":
            return

        try:
            for symbol, position in list(self.positions.items()):
                trailing = position.get("trailing_stop")
                if trailing is None:
                    continue

                # Fetch latest ticker
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    last_price = Decimal(str(ticker.get("last") or ticker.get("close") or 0))
                except Exception:
                    # If fetching fails, skip this check
                    continue

                # If long position and price <= trailing, trigger market sell
                if position.get("amount", Decimal(0)) > 0 and last_price <= Decimal(str(trailing)):
                    # Construct a minimal signal dict compatible with _execute_live_order
                    sell_signal = {
                        "symbol": symbol,
                        "order_type": "market",
                        "side": "sell",
                        "amount": float(position["amount"]),
                        "price": None,
                        "params": {"auto_trailing_stop_sell": True},
                    }
                    try:
                        await self._execute_live_order(sell_signal)
                        logger.info(f"Auto-executed trailing stop sell for {symbol} at {last_price}")
                    except Exception:
                        logger.exception(f"Failed to auto-execute trailing stop for {symbol}")
        except Exception:
            logger.exception("Error evaluating trailing stops")

    async def _estimate_trend_strength(self, symbol: str, timeframe: str = "1h", lookback: int = 20) -> float:
        """
        Estimate trend strength for a symbol using recent OHLCV data.

        Returns a float in [0.0, 1.0] where higher means stronger trend in the
        direction of the recent price movement. For live mode this will attempt
        to fetch OHLCV from the exchange; otherwise returns 0.0.

        This is a lightweight heuristic: (last_price - first_price) / first_price,
        clamped to [-1,1] and returned as absolute value in [0,1].
        """
        try:
            # Only attempt when exchange available (live)
            if not self.exchange:
                return 0.0

            # Try to fetch OHLCV; some exchanges require timeframe support
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
            except Exception:
                # fallback to ticker-based simple check
                ticker = await self.exchange.fetch_ticker(symbol)
                last = float(ticker.get("last") or ticker.get("close") or 0.0)
                return 0.0 if last == 0 else 0.0

            if not ohlcv:
                return 0.0

            # ohlcv is list of [ts, open, high, low, close, volume]
            closes = [c[4] for c in ohlcv if len(c) > 4 and c[4] is not None]
            if len(closes) < 2:
                return 0.0

            first = float(closes[0])
            last = float(closes[-1])
            if first == 0:
                return 0.0

            raw = (last - first) / abs(first)
            # Normalize to 0..1
            strength = max(0.0, min(1.0, abs(raw)))
            return float(strength)
        except Exception:
            return 0.0

    async def _attempt_reentry(
        self,
        symbol: str,
        pnl: Decimal,
        last_price: Optional[float],
        reentry_fraction: Decimal,
        max_reentries: int = 1,
    ) -> None:
        """
        Attempt profit-based re-entry using a fraction of realized profit.

        Args:
            symbol: Trading pair symbol to re-enter (e.g., 'BTC/USDT')
            pnl: Decimal profit amount realized from prior trade
            last_price: Last executed price for the trade (may be None)
            reentry_fraction: Fraction of pnl to allocate for re-entry (0..1)
            max_reentries: Maximum number of re-entry attempts
        """
        try:
            # Determine available capital to deploy (fraction of pnl)
            capital = _safe_quantize(Decimal(pnl) * Decimal(reentry_fraction))
            if capital <= 0:
                logger.debug("Re-entry skipped: capital <= 0")
                return

            # Cap capital using max_position_size * balance if available
            try:
                balance = await self.get_balance()
                max_frac = Decimal(str(self.risk_config.get("max_position_size", 0.3)))
                max_allowed = _safe_quantize(max_frac * balance)
                if capital > max_allowed:
                    capital = max_allowed
            except Exception:
                # If balance fetch fails, proceed with computed capital
                pass

            # Determine entry price (use last_price or fetch ticker)
            price = None
            if last_price is not None:
                price = float(last_price)
            else:
                try:
                    if self.exchange:
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = float(ticker.get("last") or ticker.get("close") or 0.0)
                except Exception:
                    price = None

            if not price or price <= 0:
                logger.debug("Re-entry skipped: no valid price available")
                return

            # Compute amount to buy
            amount = float(capital / Decimal(str(price)))
            if amount <= 0:
                logger.debug("Re-entry skipped: computed amount <= 0")
                return

            # Attempt re-entry up to max_reentries times with backoff
            attempt = 0
            delay = 1.0
            while attempt < max_reentries:
                attempt += 1
                try:
                    signal = {
                        "symbol": symbol,
                        "order_type": "market",
                        "side": "buy",
                        "amount": amount,
                        "price": None,
                        "params": {"reentry_from_profit": True},
                    }
                    await self.execute_order(signal)
                    logger.info(
                        f"Profit-based re-entry executed for {symbol}: capital={capital}, amount={amount}"
                    )
                    return
                except Exception:
                    logger.exception(
                        f"Re-entry attempt {attempt} failed for {symbol}; retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    delay *= 2.0

            logger.warning(f"All re-entry attempts failed for {symbol}")
        except Exception:
            logger.exception("Unhandled error in _attempt_reentry")

    async def get_active_order_count(self) -> int:
        """Get count of active/open orders."""
        return len(self.open_orders)

    async def get_open_position_count(self) -> int:
        """Get count of open positions."""
        return len(self.positions)

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self.exchange:
            await self.exchange.close()
            self.exchange = None
