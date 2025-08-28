""" 
core/order_manager.py

Handles order execution, tracking, and management across all trading modes.
Implements live trading, paper trading, and backtesting order handling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

import ccxt.async_support as ccxt
from ccxt.base.errors import NetworkError, ExchangeError

from utils.logger import TradeLogger
from utils.config_loader import ConfigLoader
from core.types import OrderType, OrderStatus
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
    timestamp: int = 0
    params: Dict = None


class OrderManager:
    """Manages order execution and tracking across all trading modes."""

    def __init__(self, config: Dict, mode: str):
        """Initialize the OrderManager."""
        self.config = config['order']
        self.risk_config = config['risk']
        self.mode = mode
        self.exchange: Optional[ccxt.Exchange] = None
        self.paper_balance = Decimal(config['paper']['initial_balance'])
        self.open_orders: Dict[str, Order] = {}
        self.closed_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict] = {}
        self.trade_count = 0

        # Initialize exchange for live trading
        if self.mode == 'live':
            self._initialize_exchange()

    def _initialize_exchange(self) -> None:
        """Initialize the exchange connection."""
        exchange_config = {
            'apiKey': self.config['exchange']['api_key'],
            'secret': self.config['exchange']['api_secret'],
            'password': self.config['exchange']['api_passphrase'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        }
        exchange_class = getattr(ccxt, self.config['exchange']['name'])
        self.exchange = exchange_class(exchange_config)

    async def execute_order(self, signal: TradingSignal) -> Optional[Dict]:
        """
        Execute an order based on the trading signal.
        
        Args:
            signal: TradingSignal containing order details
            
        Returns:
            Dictionary containing order execution results
        """
        try:
            if self.mode == 'backtest':
                return await self._execute_backtest_order(signal)
            elif self.mode == 'paper':
                return await self._execute_paper_order(signal)
            else:  # live
                return await self._execute_live_order(signal)
        except Exception as e:
            logger.error(f"Order execution failed: {str(e)}", exc_info=True)
            trade_logger.log_failed_order(signal, str(e))
            return None

    async def _execute_live_order(self, signal: TradingSignal) -> Dict:
        """
        Execute a live order on the exchange.
        
        Args:
            signal: TradingSignal containing order details
            
        Returns:
            Dictionary containing order execution details
        """
        if not self.exchange:
            raise RuntimeError("Exchange not initialized for live trading")

        order_params = {
            'symbol': signal.symbol,
            'type': signal.order_type.value,
            'side': signal.side,
            'amount': float(signal.amount),
            'price': float(signal.price) if signal.price else None,
            'params': signal.params or {}
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

    async def _execute_paper_order(self, signal: TradingSignal) -> Dict:
        """
        Simulate order execution for paper trading.
        
        Args:
            signal: TradingSignal containing order details
            
        Returns:
            Dictionary containing simulated order execution details
        """
        # Calculate fees and slippage
        fee = self._calculate_fee(signal)
        executed_price = self._apply_slippage(signal)
        
        # Calculate order cost
        cost = Decimal(signal.amount) * Decimal(executed_price)
        total_cost = cost + fee if signal.side == 'buy' else cost - fee
        
        # Check balance
        if signal.side == 'buy' and total_cost > self.paper_balance:
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
            fee={'cost': float(fee), 'currency': self.config['base_currency']},
            timestamp=int(time.time() * 1000)
        )

        # Update paper balance
        if signal.side == 'buy':
            self.paper_balance -= total_cost
        else:
            self.paper_balance += total_cost

        # Process the order
        processed_order = await self._process_order(order)
        trade_logger.log_order(processed_order, self.mode)
        return processed_order

    async def _execute_backtest_order(self, signal: TradingSignal) -> Dict:
        """
        Simulate order execution for backtesting.
        
        Args:
            signal: TradingSignal containing order details
            
        Returns:
            Dictionary containing backtest order execution details
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
            fee={'cost': float(fee), 'currency': self.config['base_currency']},
            timestamp=signal.timestamp
        )

        # Process the order (no balance tracking in backtest)
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
            id=str(response['id']),
            symbol=response['symbol'],
            type=OrderType(response['type']),
            side=response['side'],
            amount=Decimal(str(response['amount'])),
            price=Decimal(str(response['price'])) if response['price'] else None,
            status=OrderStatus(response['status']),
            filled=Decimal(str(response['filled'])),
            remaining=Decimal(str(response['remaining'])),
            cost=Decimal(str(response['cost'])) if response['cost'] else Decimal(0),
            fee=response.get('fee'),
            timestamp=response['timestamp'],
            params=response.get('params')
        )

    async def _process_order(self, order: Order) -> Dict:
        """
        Process an order after execution (live or simulated).
        
        Args:
            order: The executed order
            
        Returns:
            Dictionary with processed order details including PnL
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
        pnl = self._calculate_pnl(order) if order.side == 'sell' else None

        return {
            'id': order.id,
            'symbol': order.symbol,
            'type': order.type.value,
            'side': order.side,
            'amount': float(order.amount),
            'price': float(order.price) if order.price else None,
            'status': order.status.value,
            'cost': float(order.cost),
            'fee': order.fee,
            'timestamp': order.timestamp,
            'pnl': pnl,
            'mode': self.mode
        }

    def _update_positions(self, order: Order) -> None:
        """Update position tracking based on filled orders."""
        position = self.positions.get(order.symbol, {
            'amount': Decimal(0),
            'entry_price': Decimal(0),
            'entry_cost': Decimal(0)
        })

        if order.side == 'buy':
            new_amount = position['amount'] + order.filled
            new_cost = position['entry_cost'] + order.cost
            position.update({
                'amount': new_amount,
                'entry_price': new_cost / new_amount if new_amount > 0 else Decimal(0),
                'entry_cost': new_cost
            })
        else:
            position['amount'] -= order.filled
            if position['amount'] <= 0:
                del self.positions[order.symbol]
            else:
                self.positions[order.symbol] = position

    def _calculate_pnl(self, order: Order) -> Optional[float]:
        """Calculate PnL for a sell order."""
        if order.side != 'sell' or order.symbol not in self.positions:
            return None

        position = self.positions[order.symbol]
        entry_value = position['entry_price'] * order.filled
        exit_value = order.price * order.filled
        gross_pnl = exit_value - entry_value
        fee = Decimal(order.fee['cost']) if order.fee else Decimal(0)
        net_pnl = gross_pnl - fee

        return float(net_pnl)

    def _calculate_fee(self, signal: TradingSignal) -> Decimal:
        """Calculate trading fee based on config."""
        fee_rate = Decimal(self.config['trade_fee'])
        return Decimal(signal.amount) * fee_rate

    def _apply_slippage(self, signal: TradingSignal) -> float:
        """Apply simulated slippage to order price."""
        slippage = Decimal(self.config['slippage'])
        if signal.side == 'buy':
            return float(Decimal(signal.price) * (1 + slippage))
        else:
            return float(Decimal(signal.price) * (1 - slippage))

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.mode != 'live':
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
        if self.mode != 'live':
            return

        try:
            open_orders = list(self.open_orders.keys())
            for order_id in open_orders:
                await self.cancel_order(order_id)
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {str(e)}")

    async def get_balance(self) -> Decimal:
        """Get current account balance."""
        if self.mode == 'live':
            balance = await self.exchange.fetch_balance()
            return Decimal(str(balance['total'][self.config['base_currency']]))
        elif self.mode == 'paper':
            return self.paper_balance
        else:
            return Decimal(0)  # Backtest doesn't track balance

    async def get_equity(self) -> Decimal:
        """Get current account equity (balance + unrealized PnL)."""
        balance = await self.get_balance()
        
        if self.mode == 'live':
            # For live trading, we'd calculate unrealized PnL from open positions
            # This is a simplified version - real implementation would fetch current prices
            unrealized = Decimal(0)
            for symbol, position in self.positions.items():
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = Decimal(str(ticker['last']))
                unrealized += (current_price - position['entry_price']) * position['amount']
            
            return balance + unrealized
        else:
            return balance  # Paper/backtest doesn't track unrealized PnL in this simplified version

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
