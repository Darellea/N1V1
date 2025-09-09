"""
Execution Validator

Pre-trade validation for execution safety and compliance.
"""

import logging
from typing import Dict, Any, Optional
from decimal import Decimal

from core.contracts import TradingSignal
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class ExecutionValidator:
    """
    Validates trading signals before execution to ensure safety and compliance.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.enabled = self.config.get('enabled', True)
        self.check_balance = self.config.get('check_balance', True)
        self.check_slippage = self.config.get('check_slippage', True)
        self.max_slippage_pct = Decimal(str(self.config.get('max_slippage_pct', 0.02)))

        # Exchange constraints
        self.min_order_size = Decimal(str(self.config.get('min_order_size', 0.000001)))
        self.max_order_size = Decimal(str(self.config.get('max_order_size', 1000000)))
        self.tick_size = Decimal(str(self.config.get('tick_size', 0.00000001)))
        self.lot_size = Decimal(str(self.config.get('lot_size', 0.00000001)))

        self.logger.info("ExecutionValidator initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'check_balance': True,
            'check_slippage': True,
            'max_slippage_pct': 0.02,  # 2%
            'min_order_size': 0.000001,
            'max_order_size': 1000000,
            'tick_size': 0.00000001,
            'lot_size': 0.00000001
        }

    async def validate_signal(self, signal: TradingSignal, context: Dict[str, Any]) -> bool:
        """
        Validate a trading signal for execution.

        Args:
            signal: Trading signal to validate
            context: Market context information

        Returns:
            True if signal passes all validations
        """
        if not self.enabled:
            return True

        try:
            # Basic signal validation
            if not self._validate_basic_signal(signal):
                return False

            # Order size validation
            if not self._validate_order_size(signal):
                return False

            # Price validation
            if not self._validate_price(signal, context):
                return False

            # Balance validation
            if self.check_balance:
                if not await self._validate_balance(signal, context):
                    return False

            # Slippage validation
            if self.check_slippage:
                if not self._validate_slippage(signal, context):
                    return False

            # Exchange-specific validation
            if not self._validate_exchange_constraints(signal, context):
                return False

            self.logger.debug(f"Signal validation passed for {signal.symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error during signal validation: {e}")
            trade_logger.performance("Validation Error", {
                'symbol': signal.symbol,
                'error': str(e),
                'signal_amount': float(signal.amount)
            })
            return False

    def _validate_basic_signal(self, signal: TradingSignal) -> bool:
        """
        Validate basic signal properties.

        Args:
            signal: Trading signal

        Returns:
            True if basic validation passes
        """
        # Check required fields
        if not signal.symbol:
            self._log_validation_error(signal, "missing_symbol", "Symbol is required")
            return False

        if not signal.amount or signal.amount <= 0:
            self._log_validation_error(signal, "invalid_amount", f"Invalid amount: {signal.amount}")
            return False

        if not signal.signal_type:
            self._log_validation_error(signal, "missing_signal_type", "Signal type is required")
            return False

        if not signal.order_type:
            self._log_validation_error(signal, "missing_order_type", "Order type is required")
            return False

        return True

    def _validate_order_size(self, signal: TradingSignal) -> bool:
        """
        Validate order size constraints.

        Args:
            signal: Trading signal

        Returns:
            True if order size is valid
        """
        amount = signal.amount

        # Check minimum order size
        if amount < self.min_order_size:
            self._log_validation_error(signal, "order_too_small",
                                     f"Order size {amount} below minimum {self.min_order_size}")
            return False

        # Check maximum order size
        if amount > self.max_order_size:
            self._log_validation_error(signal, "order_too_large",
                                     f"Order size {amount} above maximum {self.max_order_size}")
            return False

        # Check lot size (must be multiple of lot size)
        # Use Decimal remainder calculation for precision
        # Handle the case where lot_size is 0 to avoid division by zero
        if self.lot_size == Decimal('0'):
            self._log_validation_error(signal, "invalid_lot_size_config", "Lot size cannot be zero")
            return False
            
        remainder = amount % self.lot_size
        # Use tolerance for floating point precision issues with Decimal
        if abs(remainder) > Decimal('1e-12') and abs(remainder - self.lot_size) > Decimal('1e-12'):
            self._log_validation_error(signal, "invalid_lot_size",
                                     f"Order size {amount} not multiple of lot size {self.lot_size}")
            return False

        return True

    def _validate_price(self, signal: TradingSignal, context: Dict[str, Any]) -> bool:
        """
        Validate price parameters.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            True if price validation passes
        """
        # For limit orders, price is required
        if signal.order_type and signal.order_type.value == 'limit':
            if not signal.price or signal.price <= 0:
                self._log_validation_error(signal, "missing_limit_price", "Limit price required for limit orders")
                return False

            # Check tick size
            if signal.price % self.tick_size != 0:
                self._log_validation_error(signal, "invalid_tick_size",
                                         f"Price {signal.price} not multiple of tick size {self.tick_size}")
                return False

        # Check price against market price for reasonableness
        market_price = context.get('market_price') or signal.current_price
        if market_price and signal.price:
            price_diff_pct = abs(signal.price - market_price) / market_price
            max_reasonable_diff = Decimal('0.1')  # 10% max difference

            if price_diff_pct > max_reasonable_diff:
                self._log_validation_error(signal, "price_too_far_from_market",
                                         f"Price {signal.price} too far from market price {market_price}")
                return False

        return True

    async def _validate_balance(self, signal: TradingSignal, context: Dict[str, Any]) -> bool:
        """
        Validate account balance for the order.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            True if balance is sufficient
        """
        # Get current balance (would be fetched from exchange/portfolio manager)
        balance = context.get('account_balance', Decimal('10000'))  # Default mock balance

        # signal.amount is the quantity of the asset, not quote currency
        market_price = context.get('market_price')
        if market_price is None:
            market_price = signal.current_price
        if market_price is None or market_price <= Decimal('0'):
            self._log_validation_error(signal, "invalid_market_price", "Market price is missing or invalid")
            return False

        slippage_pct = context.get('expected_slippage_pct', Decimal('0.001'))
        
        # Calculate total cost: quantity * market_price * (1 + slippage)
        total_cost = signal.amount * market_price * (1 + slippage_pct)

        # Add buffer for fees (estimated 0.1%)
        fee_buffer = total_cost * Decimal('0.001')
        total_required = total_cost + fee_buffer

        if balance < total_required:
            self._log_validation_error(signal, "insufficient_balance",
                                     f"Required: {total_required:.2f}, Available: {balance:.2f}")
            return False

        return True

    def _validate_slippage(self, signal: TradingSignal, context: Dict[str, Any]) -> bool:
        """
        Validate expected slippage.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            True if slippage is within acceptable range
        """
        expected_slippage = context.get('expected_slippage_pct', Decimal('0.001'))  # Default 0.1%

        if expected_slippage > self.max_slippage_pct:
            self._log_validation_error(signal, "excessive_slippage",
                                     f"Expected slippage {expected_slippage:.4f} exceeds maximum {self.max_slippage_pct:.4f}")
            return False

        return True

    def _validate_exchange_constraints(self, signal: TradingSignal, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate exchange-specific constraints.

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            True if exchange constraints are satisfied
        """
        # Add exchange-specific validations here
        # For example: trading hours, maintenance windows, symbol availability, etc.

        # Check if symbol is tradable
        tradable_symbols = self.config.get('tradable_symbols', [])
        if tradable_symbols and signal.symbol not in tradable_symbols:
            self._log_validation_error(signal, "symbol_not_tradable",
                                     f"Symbol {signal.symbol} not in tradable symbols list")
            return False

        # Check trading hours (simplified example)
        current_hour = (context or {}).get('current_hour', 12)  # Default midday
        if not (9 <= current_hour <= 16):  # Example trading hours
            self._log_validation_error(signal, "outside_trading_hours",
                                     f"Current hour {current_hour} outside trading hours 9-16")
            return False

        return True

    def _log_validation_error(self, signal: TradingSignal, error_code: str, message: str) -> None:
        """
        Log validation error.

        Args:
            signal: Trading signal that failed validation
            error_code: Error code identifier
            message: Error message
        """
        self.logger.warning(f"Validation failed for {signal.symbol}: {error_code} - {message}")

        trade_logger.performance("Validation Failed", {
            'symbol': signal.symbol,
            'error_code': error_code,
            'error_message': message,
            'signal_amount': float(signal.amount),
            'signal_type': signal.signal_type.value if signal.signal_type else 'unknown'
        })

    def get_validation_rules(self) -> Dict[str, Any]:
        """
        Get current validation rules and constraints.

        Returns:
            Dictionary of validation rules
        """
        return {
            'enabled': self.enabled,
            'check_balance': self.check_balance,
            'check_slippage': self.check_slippage,
            'max_slippage_pct': float(self.max_slippage_pct),
            'min_order_size': float(self.min_order_size),
            'max_order_size': float(self.max_order_size),
            'tick_size': float(self.tick_size),
            'lot_size': float(self.lot_size)
        }

    def update_validation_rules(self, rules: Dict[str, Any]) -> None:
        """
        Update validation rules dynamically.

        Args:
            rules: New validation rules
        """
        self.config.update(rules)

        # Update instance variables
        self.enabled = rules.get('enabled', self.enabled)
        self.check_balance = rules.get('check_balance', self.check_balance)
        self.check_slippage = rules.get('check_slippage', self.check_slippage)
        self.max_slippage_pct = Decimal(str(rules.get('max_slippage_pct', self.max_slippage_pct)))
        self.min_order_size = Decimal(str(rules.get('min_order_size', self.min_order_size)))
        self.max_order_size = Decimal(str(rules.get('max_order_size', self.max_order_size)))
        self.tick_size = Decimal(str(rules.get('tick_size', self.tick_size)))
        self.lot_size = Decimal(str(rules.get('lot_size', self.lot_size)))

        self.logger.info("Validation rules updated")
