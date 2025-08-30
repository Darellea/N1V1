"""
core/management/portfolio_manager.py

Manages portfolio-level state including per-pair balances and allocations.
"""

import logging
from decimal import Decimal
from typing import Dict, List, Optional

from risk.risk_manager import _safe_quantize

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages portfolio-level state including per-pair balances and allocations."""

    def __init__(self) -> None:
        """Initialize the PortfolioManager."""
        self.portfolio_mode: bool = False
        self.pairs: List[str] = []
        self.pair_allocation: Optional[Dict[str, float]] = None
        self.paper_balances: Dict[str, Decimal] = {}
        self.paper_balance: Decimal = Decimal("0")

    def set_initial_balance(self, initial_balance: Optional[Decimal]) -> None:
        """Set the initial paper trading balance.

        Args:
            initial_balance: Initial balance amount, or None for default
        """
        try:
            self.paper_balance = Decimal(initial_balance) if initial_balance is not None else Decimal("0")
        except Exception:
            self.paper_balance = Decimal("0")

    def initialize_portfolio(self, pairs: List[str], portfolio_mode: bool, 
                           allocation: Optional[Dict[str, float]] = None) -> None:
        """
        Initialize per-pair portfolio state.

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
                for symbol in self.pairs:
                    frac = Decimal(str(self.pair_allocation.get(symbol, 0)))
                    self.paper_balances[symbol] = _safe_quantize(total * frac)
            else:
                per = _safe_quantize(total / Decimal(len(self.pairs)))
                for symbol in self.pairs:
                    self.paper_balances[symbol] = per
        except Exception:
            logger.exception("Failed to initialize portfolio")
            return

    def update_paper_balance(self, symbol: str, amount: Decimal) -> None:
        """
        Update paper balance for a specific symbol.

        Args:
            symbol: Trading symbol
            amount: Amount to add/subtract (positive for gains, negative for losses)
        """
        if self.portfolio_mode and symbol:
            current = self.paper_balances.get(symbol, Decimal("0"))
            self.paper_balances[symbol] = _safe_quantize(current + amount)
        else:
            self.paper_balance = _safe_quantize(self.paper_balance + amount)

    def get_balance(self) -> Decimal:
        """Get current portfolio balance."""
        if self.portfolio_mode and self.paper_balances:
            # Aggregate per-pair paper balances
            total = sum([float(v) for v in self.paper_balances.values()])
            return Decimal(str(total))
        return self.paper_balance

    def get_symbol_balance(self, symbol: str) -> Decimal:
        """Get balance for a specific symbol."""
        if self.portfolio_mode and symbol in self.paper_balances:
            return self.paper_balances[symbol]
        return self.paper_balance
