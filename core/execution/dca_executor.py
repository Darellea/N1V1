"""
DCA Executor

Executes orders using Dollar-Cost Averaging strategy.
"""

import logging
import asyncio
import uuid
import time
from typing import Dict, Any, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta

from .base_executor import BaseExecutor
from core.contracts import TradingSignal
from core.types.order_types import Order, OrderStatus, OrderType
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class DCAExecutor(BaseExecutor):
    """
    Dollar-Cost Averaging executor.

    Executes orders by scaling into positions over time with configurable intervals.
    Useful for reducing entry risk and averaging purchase prices.
    """

    def __init__(self, config: Dict[str, Any], exchange_api=None):
        """
        Initialize the DCA executor.

        Args:
            config: Configuration dictionary
            exchange_api: Exchange API instance for order execution
        """
        super().__init__(config)
        self.exchange_api = exchange_api

        # Configuration
        self.interval_minutes = config.get("interval_minutes", 60)
        self.parts = config.get("parts", 5)
        self.fallback_mode = config.get("fallback_mode", "market")

        # Calculate timing
        self.interval_seconds = self.interval_minutes * 60

        self.logger.info(f"DCAExecutor initialized: interval={self.interval_minutes}min, "
                        f"parts={self.parts}")

        # Track DCA sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def execute_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute an order using DCA strategy.

        Args:
            signal: Trading signal to execute

        Returns:
            List of executed orders (may be empty if session started)
        """
        if not self._validate_signal(signal):
            return []

        # Check if this is a continuation of existing DCA session
        session_id = self._get_session_id(signal)
        if session_id in self.active_sessions:
            return await self._continue_dca_session(session_id, signal)

        # Start new DCA session
        return await self._start_dca_session(signal)

    async def _start_dca_session(self, signal: TradingSignal) -> List[Order]:
        """
        Start a new DCA session.

        Args:
            signal: Trading signal to execute

        Returns:
            List containing the first order
        """
        session_id = str(uuid.uuid4())

        self.logger.info(f"Starting DCA session: {signal.symbol} {signal.amount} "
                        f"over {self.parts} parts every {self.interval_minutes} minutes")

        trade_logger.trade("DCA Session Start", {
            "symbol": signal.symbol,
            "total_amount": float(signal.amount),
            "parts": self.parts,
            "interval_minutes": self.interval_minutes,
            "session_id": session_id
        })

        # Split the order into parts
        split_amounts = await self.split_order(signal.amount, self.parts)

        # Create DCA session
        session = {
            'session_id': session_id,
            'symbol': signal.symbol,
            'total_amount': signal.amount,
            'remaining_amount': signal.amount,
            'parts': self.parts,
            'executed_parts': 0,
            'split_amounts': split_amounts,
            'start_time': time.time(),
            'next_execution': time.time(),  # Execute first part immediately
            'executed_orders': [],
            'parent_signal': signal
        }

        self.active_sessions[session_id] = session

        # Execute first part immediately
        return await self._execute_dca_part(session_id, 0)

    async def _continue_dca_session(self, session_id: str, signal: TradingSignal) -> List[Order]:
        """
        Continue an existing DCA session.

        Args:
            session_id: DCA session ID
            signal: New signal (may be a trigger to execute next part)

        Returns:
            List of executed orders
        """
        session = self.active_sessions.get(session_id)
        if not session:
            self.logger.error(f"DCA session {session_id} not found")
            return []

        current_time = time.time()

        # Check if it's time for next execution
        if current_time >= session['next_execution']:
            next_part = session['executed_parts']
            if next_part < session['parts']:
                return await self._execute_dca_part(session_id, next_part)
            else:
                # Session completed
                self.logger.info(f"DCA session {session_id} already completed")
                return []
        else:
            # Not yet time for next execution
            remaining_time = session['next_execution'] - current_time
            self.logger.debug(f"DCA session {session_id}: {remaining_time:.0f}s until next execution")
            return []

    async def _execute_dca_part(self, session_id: str, part_index: int) -> List[Order]:
        """
        Execute a specific part of DCA session.

        Args:
            session_id: DCA session ID
            part_index: Index of the part to execute

        Returns:
            List containing the executed order
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return []

        if part_index >= len(session['split_amounts']):
            self.logger.warning(f"DCA part {part_index} out of range for session {session_id}")
            return []

        amount = session['split_amounts'][part_index]
        parent_signal = session['parent_signal']

        # Create child order
        child_signal = self._create_child_order(parent_signal, amount, session_id, part_index + 1)
        child_signal.metadata.update({
            'dca_session_id': session_id,
            'dca_part': part_index + 1,
            'dca_total_parts': session['parts'],
            'dca_interval_minutes': self.interval_minutes
        })

        # Execute the order
        executed_orders = await self._execute_single_order(child_signal)

        if executed_orders:
            # Update session
            session['executed_parts'] += 1
            session['executed_orders'].extend([order.id for order in executed_orders])
            session['remaining_amount'] -= amount

            # Schedule next execution
            if session['executed_parts'] < session['parts']:
                session['next_execution'] = time.time() + self.interval_seconds
                self.logger.info(f"DCA session {session_id}: part {part_index + 1}/{session['parts']} executed, "
                               f"next in {self.interval_minutes} minutes")
            else:
                # Session completed
                self._complete_dca_session(session_id)

            trade_logger.trade("DCA Part Executed", {
                "session_id": session_id,
                "part": part_index + 1,
                "total_parts": session['parts'],
                "amount": float(amount),
                "remaining": float(session['remaining_amount'])
            })

        return executed_orders

    def _complete_dca_session(self, session_id: str) -> None:
        """
        Complete a DCA session.

        Args:
            session_id: DCA session ID
        """
        session = self.active_sessions.get(session_id)
        if session:
            duration = time.time() - session['start_time']
            self.logger.info(f"DCA session {session_id} completed: "
                           f"{session['executed_parts']}/{session['parts']} parts "
                           f"in {duration:.1f}s")

            trade_logger.trade("DCA Session Complete", {
                "session_id": session_id,
                "parts_executed": session['executed_parts'],
                "total_parts": session['parts'],
                "duration_seconds": duration,
                "symbol": session['symbol']
            })

            # Clean up session
            del self.active_sessions[session_id]

    def _get_session_id(self, signal: TradingSignal) -> Optional[str]:
        """
        Get DCA session ID from signal metadata.

        Args:
            signal: Trading signal

        Returns:
            Session ID if found, None otherwise
        """
        if signal.metadata and 'dca_session_id' in signal.metadata:
            return signal.metadata['dca_session_id']
        return None

    async def _execute_single_order(self, signal: TradingSignal) -> List[Order]:
        """
        Execute a single order.

        Args:
            signal: Trading signal to execute

        Returns:
            List containing the executed order
        """
        try:
            if self.exchange_api:
                # Execute via exchange API
                order_response = await self._place_order_via_api(signal)
                order = self._parse_order_response(order_response)
            else:
                # Create mock order for testing/paper trading
                order = self._create_mock_order(signal)

            return [order] if order else []

        except Exception as e:
            self.logger.error(f"Failed to execute DCA order part: {e}")
            return []

    async def _place_order_via_api(self, signal: TradingSignal) -> Dict[str, Any]:
        """
        Place order via exchange API.

        Args:
            signal: Trading signal

        Returns:
            Order response from exchange
        """
        if not self.exchange_api:
            raise ValueError("Exchange API not configured")

        # Determine order parameters
        side = "buy" if signal.signal_type.value == "ENTRY_LONG" else "sell"
        order_type = signal.order_type.value if signal.order_type else "market"

        params = {
            'symbol': signal.symbol,
            'type': order_type,
            'side': side,
            'amount': float(signal.amount)
        }

        if signal.price and order_type == "limit":
            params['price'] = float(signal.price)

        # Add DCA metadata
        if signal.metadata:
            params['metadata'] = signal.metadata

        # Place the order
        response = await self.exchange_api.create_order(**params)
        return response

    def _parse_order_response(self, response: Dict[str, Any]) -> Order:
        """
        Parse exchange order response.

        Args:
            response: Raw order response

        Returns:
            Parsed Order object
        """
        return Order(
            id=str(response.get('id', '')),
            symbol=response.get('symbol', ''),
            type=OrderType(response.get('type', 'market')),
            side=response.get('side', ''),
            amount=Decimal(str(response.get('amount', 0))),
            price=Decimal(str(response.get('price', 0))) if response.get('price') else None,
            status=OrderStatus(response.get('status', 'open')),
            timestamp=response.get('timestamp', 0),
            filled=Decimal(str(response.get('filled', 0))),
            remaining=Decimal(str(response.get('remaining', 0))),
            cost=Decimal(str(response.get('cost', 0))),
            fee=response.get('fee')
        )

    def _create_mock_order(self, signal: TradingSignal) -> Order:
        """
        Create a mock order for testing/paper trading.

        Args:
            signal: Trading signal

        Returns:
            Mock Order object
        """
        return Order(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            type=signal.order_type or OrderType.MARKET,
            side="buy" if signal.signal_type.value == "ENTRY_LONG" else "sell",
            amount=signal.amount,
            price=signal.price,
            status=OrderStatus.FILLED,
            timestamp=signal.timestamp,
            filled=signal.amount,
            remaining=Decimal(0),
            cost=signal.amount * (signal.price or Decimal(1)),
            fee={'cost': Decimal(0), 'currency': 'USD'}
        )

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if cancellation was successful
        """
        try:
            if self.exchange_api:
                await self.exchange_api.cancel_order(order_id)
            self.logger.info(f"Cancelled DCA order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel DCA order {order_id}: {e}")
            return False

    async def cancel_dca_session(self, session_id: str) -> bool:
        """
        Cancel an entire DCA session.

        Args:
            session_id: DCA session ID

        Returns:
            True if cancellation was successful
        """
        session = self.active_sessions.get(session_id)
        if not session:
            self.logger.warning(f"DCA session {session_id} not found")
            return False

        # Cancel all pending orders in the session
        cancelled_count = 0
        for order_id in session.get('executed_orders', []):
            try:
                await self.cancel_order(order_id)
                cancelled_count += 1
            except Exception as e:
                self.logger.error(f"Failed to cancel order {order_id} in session {session_id}: {e}")

        # Remove session
        del self.active_sessions[session_id]

        self.logger.info(f"Cancelled DCA session {session_id}: {cancelled_count} orders cancelled")
        return True

    def get_dca_sessions(self) -> List[Dict[str, Any]]:
        """
        Get information about all active DCA sessions.

        Returns:
            List of DCA session information
        """
        sessions = []
        for session_id, session in self.active_sessions.items():
            sessions.append({
                'session_id': session_id,
                'symbol': session['symbol'],
                'total_amount': float(session['total_amount']),
                'remaining_amount': float(session['remaining_amount']),
                'parts': session['parts'],
                'executed_parts': session['executed_parts'],
                'progress': session['executed_parts'] / session['parts'],
                'next_execution': session['next_execution'],
                'time_until_next': max(0, session['next_execution'] - time.time())
            })
        return sessions

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a specific DCA session.

        Args:
            session_id: DCA session ID

        Returns:
            Session status information
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        return {
            'session_id': session_id,
            'symbol': session['symbol'],
            'total_amount': float(session['total_amount']),
            'remaining_amount': float(session['remaining_amount']),
            'parts': session['parts'],
            'executed_parts': session['executed_parts'],
            'progress_percentage': (session['executed_parts'] / session['parts']) * 100,
            'next_execution_time': session['next_execution'],
            'seconds_until_next': max(0, session['next_execution'] - time.time()),
            'executed_orders': session['executed_orders'].copy(),
            'start_time': session['start_time'],
            'duration_so_far': time.time() - session['start_time']
        }
