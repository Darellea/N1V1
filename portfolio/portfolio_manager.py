"""
portfolio/portfolio_manager.py

Main Portfolio Manager for Multi-Asset Trading.

This module provides comprehensive portfolio management capabilities including:
- Multi-asset position tracking and capital allocation
- Dynamic asset rotation based on momentum and performance
- Adaptive rebalancing with multiple allocation schemes
- Risk management through hedging strategies
- Portfolio metrics and performance reporting
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict

import pandas as pd
import numpy as np
from decimal import Decimal

from .allocator import CapitalAllocator, EqualWeightAllocator, RiskParityAllocator, MomentumWeightAllocator
from .hedging import PortfolioHedger


@dataclass
class Position:
    """Represents a position in a specific asset."""

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Optional[Decimal] = None
    entry_time: datetime = None
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None

    def __post_init__(self):
        """Initialize position with current timestamp if not provided."""
        if self.entry_time is None:
            self.entry_time = datetime.now()

    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of the position."""
        if self.current_price is None:
            return Decimal('0')
        return self.quantity * self.current_price

    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    def update_pnl(self, current_price: Decimal) -> None:
        """Update unrealized P&L based on current price."""
        if current_price is not None:
            self.current_price = current_price
            entry_value = self.quantity * self.entry_price
            current_value = self.market_value
            self.unrealized_pnl = current_value - entry_value


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""

    total_value: Decimal
    total_pnl: Decimal
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_positions: int
    num_assets: int
    allocation_history: List[Dict[str, Any]]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['total_value'] = float(self.total_value)
        data['total_pnl'] = float(self.total_pnl)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PortfolioManager:
    """
    Main Portfolio Manager for Multi-Asset Trading.

    This class provides comprehensive portfolio management including:
    - Position tracking across multiple assets
    - Dynamic asset rotation based on performance
    - Adaptive rebalancing with configurable schemes
    - Risk management through hedging
    - Performance metrics and reporting
    """

    def __init__(self, config: Dict[str, Any], initial_balance: Decimal = Decimal('10000')):
        """
        Initialize Portfolio Manager.

        Args:
            config: Portfolio configuration dictionary
            initial_balance: Initial portfolio balance
        """
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Portfolio state
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.allocation_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.portfolio_history: List[PortfolioMetrics] = []
        self.last_rebalance_time = datetime.now()
        self.last_rotation_time = datetime.now()

        # Initialize components
        self._setup_components()

        # Setup logging
        self._setup_logging()

        self.logger.info(f"Portfolio Manager initialized with ${initial_balance} balance")

    def _setup_components(self) -> None:
        """Setup portfolio management components."""
        # Initialize capital allocator
        allocation_config = self.config.get('rebalancing', {})
        scheme = allocation_config.get('scheme', 'equal_weight')

        if scheme == 'equal_weight':
            self.allocator = EqualWeightAllocator()
        elif scheme == 'risk_parity':
            self.allocator = RiskParityAllocator()
        elif scheme == 'momentum_weighted':
            self.allocator = MomentumWeightAllocator()
        else:
            self.allocator = EqualWeightAllocator()

        # Initialize hedger
        hedging_config = self.config.get('hedging', {})
        self.hedger = PortfolioHedger(hedging_config) if hedging_config.get('enabled', False) else None

    def _setup_logging(self) -> None:
        """Setup logging for portfolio manager."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def update_prices(self, price_data: Dict[str, Decimal]) -> None:
        """
        Update current prices for all assets.

        Args:
            price_data: Dictionary mapping symbols to current prices
        """
        for symbol, price in price_data.items():
            if symbol in self.positions:
                self.positions[symbol].update_pnl(price)

        self.logger.debug(f"Updated prices for {len(price_data)} assets")

    def get_portfolio_value(self) -> Decimal:
        """
        Calculate total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash_balance + positions_value

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio metrics.

        Returns:
            PortfolioMetrics object with current metrics
        """
        total_value = self.get_portfolio_value()
        total_pnl = total_value - self.initial_balance
        total_return = float((total_value - self.initial_balance) / self.initial_balance)
        
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        win_rate = self._calculate_win_rate()
        num_positions = len(self.positions)
        num_assets = len(set(pos.symbol for pos in self.positions.values()))

        metrics = PortfolioMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            num_positions=num_positions,
            num_assets=num_assets,
            allocation_history=self.allocation_history[-10:],  # Last 10 allocations
            timestamp=datetime.now()
        )

        # Store in history
        self.portfolio_history.append(metrics)

        return metrics

    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio based on historical returns.
        
        Returns:
            Sharpe ratio value
        """
        if len(self.portfolio_history) > 1:
            returns = [m.total_return for m in self.portfolio_history[-30:]]  # Last 30 periods
            if len(returns) > 1 and np.std(returns) > 0:
                return np.mean(returns) / np.std(returns) * np.sqrt(252)
        return 0.0

    def _calculate_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from portfolio history.
        
        Returns:
            Maximum drawdown value
        """
        if not self.portfolio_history:
            return 0.0
            
        values = [m.total_value for m in self.portfolio_history]
        peak = max(values)
        if peak > 0:
            return (peak - min(values)) / peak
        return 0.0

    def _calculate_win_rate(self) -> float:
        """
        Calculate win rate based on profitable positions.
        
        Returns:
            Win rate as a percentage
        """
        winning_positions = sum(1 for pos in self.positions.values() if pos.total_pnl > 0)
        total_positions = len(self.positions)
        return winning_positions / total_positions if total_positions > 0 else 0.0

    def rotate_assets(self, strategy_signals: Dict[str, Any],
                     market_data: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Rotate assets based on strategy signals and market conditions.

        Args:
            strategy_signals: Dictionary of strategy signals per asset
            market_data: Historical market data for momentum calculation

        Returns:
            List of selected assets for allocation
        """
        rotation_config = self.config.get('rotation', {})
        method = rotation_config.get('method', 'momentum')
        top_n = rotation_config.get('top_n', 5)

        if method == 'momentum' and market_data is not None:
            selected_assets = self._rotate_by_momentum(market_data, top_n)
        elif method == 'signal_strength':
            selected_assets = self._rotate_by_signal_strength(strategy_signals, top_n)
        elif method == 'performance':
            selected_assets = self._rotate_by_performance(top_n)
        else:
            # Default: use all assets with signals
            selected_assets = list(strategy_signals.keys())

        # Limit to top N assets
        selected_assets = selected_assets[:top_n]

        self.last_rotation_time = datetime.now()
        self.logger.info(f"Asset rotation completed. Selected assets: {selected_assets}")

        return selected_assets

    def _rotate_by_momentum(self, market_data: pd.DataFrame, top_n: int) -> List[str]:
        """
        Rotate assets based on momentum scores.

        Args:
            market_data: Historical market data
            top_n: Number of top assets to select

        Returns:
            List of selected asset symbols
        """
        lookback_days = self.config.get('rotation', {}).get('lookback_days', 30)

        momentum_scores = {}
        for symbol in market_data.columns:
            if symbol in market_data.columns:
                try:
                    prices = market_data[symbol].dropna()
                    if len(prices) >= lookback_days:
                        # Calculate momentum as percentage change over lookback period
                        momentum = (prices.iloc[-1] - prices.iloc[-lookback_days]) / prices.iloc[-lookback_days]
                        momentum_scores[symbol] = float(momentum)
                except Exception as e:
                    self.logger.debug(f"Error calculating momentum for {symbol}: {str(e)}")
                    continue

        # Sort by momentum score (descending)
        sorted_assets = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_assets[:top_n]]

    def _rotate_by_signal_strength(self, strategy_signals: Dict[str, Any], top_n: int) -> List[str]:
        """
        Rotate assets based on signal strength.

        Args:
            strategy_signals: Dictionary of strategy signals
            top_n: Number of top assets to select

        Returns:
            List of selected asset symbols
        """
        signal_strengths = {}
        for symbol, signals in strategy_signals.items():
            if isinstance(signals, list) and signals:
                # Calculate average signal strength
                strengths = [getattr(s, 'signal_strength', 0) for s in signals if hasattr(s, 'signal_strength')]
                if strengths:
                    signal_strengths[symbol] = np.mean(strengths)

        # Sort by signal strength (descending)
        sorted_assets = sorted(signal_strengths.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_assets[:top_n]]

    def _rotate_by_performance(self, top_n: int) -> List[str]:
        """
        Rotate assets based on recent performance.

        Args:
            top_n: Number of top assets to select

        Returns:
            List of selected asset symbols
        """
        performance_scores = {}
        for symbol, position in self.positions.items():
            if position.total_pnl != 0:
                # Calculate performance as P&L percentage
                entry_value = position.quantity * position.entry_price
                performance_scores[symbol] = float(position.total_pnl / entry_value)

        # Sort by performance (descending)
        sorted_assets = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in sorted_assets[:top_n]]

    def rebalance(self, target_allocations: Dict[str, float],
                  current_prices: Optional[Dict[str, Decimal]] = None) -> Dict[str, Any]:
        """
        Rebalance portfolio to target allocations.

        Args:
            target_allocations: Dictionary mapping symbols to target allocation percentages
            current_prices: Current prices for all assets

        Returns:
            Dictionary with rebalancing results
        """
        # Check if rebalancing is needed
        rebalance_check = self._check_rebalance_needed(target_allocations)
        if not rebalance_check['should_rebalance']:
            # Return consistent structure for no-rebalance case
            return {
                "rebalanced": False,
                "actions": [],
                "current_allocations": self._get_current_allocations(),
                "target_allocations": target_allocations,
                "reason": rebalance_check.get('reason', 'no_rebalance_needed')
            }

        # Calculate and execute trades
        total_value = self.get_portfolio_value()
        trades = self._calculate_rebalance_trades(target_allocations, total_value, current_prices)
        executed_trades = self._execute_trades(trades)

        # Update allocation history and timestamp
        self._update_allocation_history(target_allocations, executed_trades)
        self.last_rebalance_time = datetime.now()
        self.logger.info(f"Portfolio rebalanced. Executed {len(executed_trades)} trades")

        return {
            "rebalanced": True,
            "actions": executed_trades,
            "trades": executed_trades,  # Keep both for compatibility
            "current_allocations": self._get_current_allocations(),
            "target_allocations": target_allocations,
            "total_value": float(total_value)
        }

    def _check_rebalance_needed(self, target_allocations: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Check if rebalancing is needed based on configuration.

        Args:
            target_allocations: Target allocation percentages (optional)

        Returns:
            Dictionary with rebalance decision
        """
        rebalance_config = self.config.get('rebalancing', {})
        mode = rebalance_config.get('mode', 'threshold')
        threshold = rebalance_config.get('threshold', 0.05)

        if mode == 'threshold':
            # Use provided target allocations or get from allocator
            allocations_to_check = target_allocations or {}
            if not allocations_to_check and hasattr(self, 'allocator') and self.allocator:
                allocations_to_check = getattr(self.allocator, 'get_target_allocations', lambda: {})()

            # If no target allocations available, assume rebalancing is needed
            if not allocations_to_check:
                return {'should_rebalance': True}

            if not self._should_rebalance_threshold(allocations_to_check, threshold):
                self.logger.info("Rebalancing not needed - allocations within threshold")
                return {'should_rebalance': False, 'reason': 'within_threshold'}

        elif mode == 'periodic':
            period_days = rebalance_config.get('period_days', 7)
            if not self._should_rebalance_periodic(period_days):
                self.logger.info("Rebalancing not needed - within periodic interval")
                return {'should_rebalance': False, 'reason': 'within_period'}

        return {'should_rebalance': True}

    def _update_allocation_history(self, target_allocations: Dict[str, float],
                                  executed_trades: List[Dict[str, Any]]) -> None:
        """
        Update allocation history with rebalancing record.

        Args:
            target_allocations: Target allocation percentages
            executed_trades: List of executed trades
        """
        allocation_record = {
            'timestamp': datetime.now(),
            'target_allocations': target_allocations,
            'current_allocations': self._get_current_allocations(),
            'trades': executed_trades
        }
        self.allocation_history.append(allocation_record)

    def _should_rebalance_threshold(self, target_allocations: Dict[str, float],
                                   threshold: float) -> bool:
        """
        Check if rebalancing is needed based on threshold deviation.

        Args:
            target_allocations: Target allocation percentages
            threshold: Maximum allowed deviation

        Returns:
            True if rebalancing is needed
        """
        # If no positions exist, don't rebalance (no existing allocation to deviate from)
        if len(self.positions) == 0:
            return False

        current_allocations = self._get_current_allocations()

        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocations.get(symbol, 0.0)
            # Skip checking assets with zero current allocation (new assets)
            if current_pct == 0.0:
                continue
            deviation = abs(current_pct - target_pct)

            if deviation > threshold:
                return True

        return False

    def _should_rebalance_periodic(self, period_days: int) -> bool:
        """
        Check if rebalancing is needed based on time period.

        Args:
            period_days: Number of days between rebalancing

        Returns:
            True if rebalancing is needed
        """
        time_since_rebalance = datetime.now() - self.last_rebalance_time
        return time_since_rebalance.days >= period_days

    def _get_current_allocations(self) -> Dict[str, float]:
        """
        Get current portfolio allocations as percentages.

        Returns:
            Dictionary mapping symbols to current allocation percentages
        """
        total_value = self.get_portfolio_value()
        if total_value == 0:
            return {}

        allocations = {}
        for symbol, position in self.positions.items():
            allocation_pct = float(position.market_value / total_value)
            allocations[symbol] = allocation_pct

        # Include cash allocation
        cash_pct = float(self.cash_balance / total_value)
        allocations['CASH'] = cash_pct

        return allocations

    def _calculate_rebalance_trades(self, target_allocations: Dict[str, float],
                                   total_value: Decimal,
                                   current_prices: Optional[Dict[str, Decimal]] = None) -> List[Dict[str, Any]]:
        """
        Calculate trades needed to achieve target allocations.

        Args:
            target_allocations: Target allocation percentages
            total_value: Total portfolio value
            current_prices: Current prices for all assets

        Returns:
            List of trade orders
        """
        trades = []
        current_allocations = self._get_current_allocations()

        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocations.get(symbol, 0.0)
            target_value = total_value * Decimal(str(target_pct))
            current_value = total_value * Decimal(str(current_pct))

            value_difference = target_value - current_value

            if abs(value_difference) > Decimal('1'):  # Minimum trade size
                # Get price for the symbol
                price = None
                if current_prices and symbol in current_prices:
                    price = current_prices[symbol]
                elif symbol in self.positions and self.positions[symbol].current_price:
                    price = self.positions[symbol].current_price

                if price is not None:
                    quantity = value_difference / price

                    trade = {
                        'symbol': symbol,
                        'side': 'buy' if quantity > 0 else 'sell',
                        'quantity': abs(float(quantity)),
                        'price': float(price),
                        'value': abs(float(value_difference)),
                        'reason': 'rebalance'
                    }
                    trades.append(trade)

        return trades

    def _execute_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute calculated trades.

        Args:
            trades: List of trade orders

        Returns:
            List of executed trades
        """
        executed_trades = []

        for trade in trades:
            try:
                # In a real implementation, this would interface with the exchange
                # For now, we'll simulate execution
                executed_trade = trade.copy()
                executed_trade['status'] = 'executed'
                executed_trade['timestamp'] = datetime.now()

                # Update portfolio state
                self._update_portfolio_from_trade(executed_trade)

                executed_trades.append(executed_trade)
                self.logger.info(f"Executed trade: {executed_trade}")

            except Exception as e:
                self.logger.error(f"Failed to execute trade {trade}: {str(e)}")
                executed_trade = trade.copy()
                executed_trade['status'] = 'failed'
                executed_trade['error'] = str(e)
                executed_trades.append(executed_trade)

        return executed_trades

    def _update_portfolio_from_trade(self, trade: Dict[str, Any]) -> None:
        """
        Update portfolio state based on executed trade.

        Args:
            trade: Executed trade details
        """
        symbol = trade['symbol']
        side = trade['side']
        quantity = Decimal(str(trade['quantity']))
        price = Decimal(str(trade['price']))
        value = Decimal(str(trade['value']))

        if side == 'buy':
            # Add to position or create new position
            if symbol in self.positions:
                existing_pos = self.positions[symbol]
                total_quantity = existing_pos.quantity + quantity
                # Weighted average price
                total_value = (existing_pos.quantity * existing_pos.entry_price) + (quantity * price)
                new_entry_price = total_value / total_quantity

                existing_pos.quantity = total_quantity
                existing_pos.entry_price = new_entry_price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price
                )

            self.cash_balance -= value

        elif side == 'sell':
            # Reduce position
            if symbol in self.positions:
                position = self.positions[symbol]
                if position.quantity >= quantity:
                    # Calculate realized P&L
                    realized_pnl = (price - position.entry_price) * quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity

                    # Remove position if fully closed
                    if position.quantity == 0:
                        del self.positions[symbol]

                    self.cash_balance += value

    def hedge_positions(self, market_conditions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Apply hedging strategy based on market conditions.

        Args:
            market_conditions: Current market conditions

        Returns:
            Hedging actions taken, or None if no hedging applied
        """
        if not self.hedger:
            return None

        hedging_actions = self.hedger.evaluate_hedging(self.positions, market_conditions)

        if hedging_actions:
            self.logger.info(f"Applying hedging actions: {hedging_actions}")
            # Execute hedging trades
            executed_hedges = self._execute_trades(hedging_actions.get('trades', []))
            hedging_actions['executed_trades'] = executed_hedges

        return hedging_actions

    def get_allocation_history(self) -> List[Dict[str, Any]]:
        """
        Get portfolio allocation history.

        Returns:
            List of allocation records
        """
        return self.allocation_history

    def export_allocation_history(self, filepath: str) -> str:
        """
        Export allocation history to CSV file.

        Args:
            filepath: Path to export file

        Returns:
            Path to exported file
        """
        import csv

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'symbol', 'allocation_pct', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in self.allocation_history:
                timestamp = record['timestamp']
                for symbol, allocation_pct in record.get('current_allocations', {}).items():
                    total_value = record.get('total_value', 0)
                    value = total_value * allocation_pct if total_value > 0 else 0

                    writer.writerow({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'allocation_pct': allocation_pct,
                        'value': value
                    })

        self.logger.info(f"Allocation history exported to {filepath}")
        return filepath

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.

        Returns:
            Dictionary with portfolio summary
        """
        metrics = self.get_portfolio_metrics()

        summary = {
            'portfolio_value': float(metrics.total_value),
            'total_pnl': float(metrics.total_pnl),
            'total_return_pct': metrics.total_return * 100,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown_pct': metrics.max_drawdown * 100,
            'win_rate_pct': metrics.win_rate * 100,
            'num_positions': metrics.num_positions,
            'num_assets': metrics.num_assets,
            'cash_balance': float(self.cash_balance),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'quantity': float(pos.quantity),
                    'entry_price': float(pos.entry_price),
                    'current_price': float(pos.current_price) if pos.current_price else None,
                    'market_value': float(pos.market_value),
                    'unrealized_pnl': float(pos.unrealized_pnl),
                    'total_pnl': float(pos.total_pnl)
                }
                for pos in self.positions.values()
            ],
            'last_rebalance': self.last_rebalance_time.isoformat(),
            'last_rotation': self.last_rotation_time.isoformat()
        }

        return summary
