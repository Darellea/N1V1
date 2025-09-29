"""
core/execution/paper_executor.py

Handles paper trading order execution and simulation.
"""

import logging
from decimal import Decimal
from typing import Dict, Any, Optional
from utils.time import now_ms

from core.types.order_types import Order, OrderStatus
from utils.logger import get_trade_logger
from utils.adapter import signal_to_dict
from risk.risk_manager import _safe_quantize

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class PaperOrderExecutor:
    """Handles paper trading order execution and simulation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the PaperOrderExecutor.

        Args:
            config: Configuration dictionary with paper trading settings
        """
        self.config = config
        self.paper_balance: Decimal = Decimal("0")
        self.paper_balances: Dict[str, Decimal] = {}
        self.trade_count: int = 0
        self.portfolio_mode: bool = False
        self.pairs: Optional[list] = None
        self.pair_allocation: Optional[Dict[str, float]] = None

    def set_initial_balance(self, initial_balance: Optional[Decimal]) -> None:
        """Set the initial paper trading balance.

        Args:
            initial_balance: Initial balance amount, or None for default
        """
        try:
            self.paper_balance = Decimal(initial_balance) if initial_balance is not None else Decimal("0")
        except Exception:
            self.paper_balance = Decimal("0")

    def set_portfolio_mode(self, portfolio_mode: bool, pairs: Optional[list] = None, 
                          allocation: Optional[Dict[str, float]] = None) -> None:
        """Configure portfolio mode settings.

        Args:
            portfolio_mode: Whether portfolio mode is enabled
            pairs: List of trading symbols
            allocation: Optional mapping symbol->fraction of total initial balance
        """
        self.portfolio_mode = bool(portfolio_mode)
        self.pairs = pairs or []
        self.pair_allocation = allocation or None

        if not self.portfolio_mode or not self.pairs:
            return

        # Determine initial capital per pair. Use allocation if provided, else equal split.
        total = self.paper_balance
        if self.pair_allocation:
            for symbol in self.pairs:
                frac = Decimal(str(self.pair_allocation.get(symbol, 0)))
                self.paper_balances[symbol] = _safe_quantize(total * frac)
        else:
            per = _safe_quantize(total / Decimal(len(self.pairs)))
            for symbol in self.pairs:
                self.paper_balances[symbol] = per

    async def execute_paper_order(self, signal: Any) -> Dict[str, Any]:
        """
        Simulate order execution for paper trading.

        Args:
            signal: Object providing order details.

        Returns:
            Dictionary containing simulated order execution details.
        """
        # Determine side
        side = getattr(signal, 'side', None)
        if side is None:
            from core.contracts import SignalType
            if signal.signal_type == SignalType.ENTRY_LONG:
                side = "buy"
            elif signal.signal_type == SignalType.ENTRY_SHORT:
                side = "sell"
            else:
                side = "buy"

        # Calculate slippage
        executed_price = self._apply_slippage(signal, side)

        # Calculate order cost
        cost = Decimal(signal.amount) * Decimal(executed_price)

        # Calculate fees
        fee = self._calculate_fee(cost)

        # Check balance
        symbol = getattr(signal, "symbol", None) or (signal.get("symbol") if isinstance(signal, dict) else None)
        if side == "buy":
            if self.portfolio_mode and symbol:
                # Check per-symbol balance in portfolio mode
                available_balance = self.paper_balances.get(symbol, self.paper_balance)
            else:
                # Check main balance in single mode
                available_balance = self.paper_balance

            if (cost + fee) > available_balance:
                raise ValueError("Insufficient balance for paper trading order")

        # Convert order_type to OrderType enum
        from core.types.order_types import OrderType
        order_type = getattr(signal, "order_type", None) or signal.get("order_type", OrderType.MARKET)
        if isinstance(order_type, str):
            try:
                order_type_enum = OrderType(order_type.lower())
            except ValueError:
                order_type_enum = OrderType.MARKET
        else:
            order_type_enum = order_type

        # Create simulated order
        order = Order(
            id=f"paper_{self.trade_count}",
            symbol=signal.symbol,
            type=order_type_enum,
            side=side,
            amount=Decimal(signal.amount),
            price=Decimal(executed_price),
            status=OrderStatus.FILLED,
            filled=Decimal(signal.amount),
            remaining=Decimal(0),
            cost=cost,
            params=getattr(signal, "params", {}) or ({"stop_loss": getattr(signal, "stop_loss", None)}),
            fee={"cost": float(fee), "currency": self.config.get("order", {}).get("base_currency", "USDT")},
            trailing_stop=(
                Decimal(str(signal.trailing_stop.get("price")))
                if getattr(signal, "trailing_stop", None)
                and isinstance(signal.trailing_stop, dict)
                and signal.trailing_stop.get("price")
                else None
            ),
            timestamp=now_ms(),
        )

        # Update paper balance (support per-pair balances when portfolio_mode enabled)
        symbol = getattr(signal, "symbol", None) or (signal.get("symbol") if isinstance(signal, dict) else None)
        if self.portfolio_mode and symbol:
            # ensure per-symbol balance exists
            bal = self.paper_balances.setdefault(symbol, Decimal(self.paper_balance))
            if side == "buy":
                bal -= (cost + fee)
            else:
                bal += (cost - fee)
            self.paper_balances[symbol] = bal.quantize(Decimal("0.01"))
        else:
            if side == "buy":
                self.paper_balance -= (cost + fee)
            else:
                self.paper_balance += (cost - fee)
            self.paper_balance = self.paper_balance.quantize(Decimal("0.01"))

        self.trade_count += 1
        return order

    def _calculate_fee(self, cost: Decimal) -> Decimal:
        """Calculate trading fee based on config.

        Args:
            cost: Order cost.

        Returns:
            Decimal fee amount.
        """
        # Get fee rate from paper config section, fallback to order section, then root
        paper_config = self.config.get("paper", {})
        fee_rate = paper_config.get("trade_fee") or self.config.get("order", {}).get("trade_fee") or self.config.get("trade_fee", "0.001")
        fee_rate = Decimal(str(fee_rate))
        return cost * fee_rate

    def _extract_price(self, signal: Any):
        for key in ("price", "current_price"):
            # Attribute style access
            if hasattr(signal, key):
                value = getattr(signal, key)
                if value is not None:
                    return value
            # Dict-style access
            if isinstance(signal, dict) and key in signal:
                return signal[key]
        raise ValueError("Signal price required")

    def _apply_slippage(self, signal: Any, side: str) -> Decimal:
        """Apply simulated slippage to order price.

        Args:
            signal: Object providing 'price' or 'current_price'.
            side: Side of the order.

        Returns:
            Adjusted price (Decimal) including slippage.
        """
        # Get slippage from paper config section, fallback to order section, then root
        paper_config = self.config.get("paper", {})
        slippage = paper_config.get("slippage") or self.config.get("order", {}).get("slippage") or self.config.get("slippage", "0.0005")
        slippage = Decimal(str(slippage))
        price = self._extract_price(signal)
        try:
            price_d = Decimal(str(price))
        except Exception:
            raise ValueError("Signal price required")
        if side == "buy":
            adjusted = price_d * (1 + slippage)
        else:
            adjusted = price_d * (1 - slippage)
        return adjusted.quantize(Decimal("0.0001"))


    def get_balance(self) -> Decimal:
        """Get current paper balance."""
        if self.portfolio_mode:
            if self.paper_balances:
                # Aggregate per-pair paper balances
                total = sum(self.paper_balances.values())
                return total
            else:
                # Portfolio mode but no balances allocated yet
                return Decimal("0")
        return self.paper_balance
