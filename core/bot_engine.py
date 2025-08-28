"""
core/bot_engine.py

The central engine that orchestrates all trading operations.
Handles the main event loop, mode switching, and module coordination.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.console import Console

from utils.logger import setup_logging
from utils.config_loader import ConfigLoader
from data.data_fetcher import DataFetcher
from strategies.base_strategy import BaseStrategy
from risk.risk_manager import RiskManager
from core.order_manager import OrderManager
from core.signal_router import SignalRouter
from notifier.discord_bot import DiscordNotifier

console = Console()
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Enum representing different trading modes."""
    LIVE = auto()
    PAPER = auto()
    BACKTEST = auto()


@dataclass
class BotState:
    """Dataclass to hold the current state of the bot."""
    running: bool = True
    paused: bool = False
    active_orders: int = 0
    open_positions: int = 0
    balance: float = 0.0
    equity: float = 0.0


class BotEngine:
    """Main trading bot engine that coordinates all components."""
    
    def __init__(self, config: Dict):
        """Initialize the bot engine with configuration."""
        self.config = config
        self.mode = TradingMode[config['environment']['mode'].upper()]
        self.state = BotState()
        
        # Core modules
        self.data_fetcher: Optional[DataFetcher] = None
        self.strategies: List[BaseStrategy] = []
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.signal_router: Optional[SignalRouter] = None
        self.notifier: Optional[DiscordNotifier] = None
        
        # Performance tracking
        self.performance_stats: Dict[str, Any] = {
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_pnl': 0.0,
            'equity_history': [],
            'returns_history': [],
            # equity_progression stores dicts with schema:
            # {'trade_id', 'timestamp', 'equity', 'pnl', 'cumulative_return'}
            'equity_progression': [],
        }
        
        # UI components
        self.live_display: Optional[Live] = None
        self.display_table: Optional[Table] = None

    async def initialize(self) -> None:
        """Initialize all components of the trading bot."""
        logger.info("Initializing BotEngine")
        
        # Load configuration
        exchange_config = self.config['exchange']
        trading_config = self.config['trading']
        
        # Initialize modules
        self.data_fetcher = DataFetcher(exchange_config)
        await self.data_fetcher.initialize()
        
        self.risk_manager = RiskManager(self.config['risk_management'])
        self.order_manager = OrderManager(trading_config, self.mode)
        # Record starting balance from config for cumulative return calculations
        try:
            self.starting_balance = float(self.config.get('trading', {}).get('initial_balance', 1000.0))
        except Exception:
            self.starting_balance = 1000.0
        self.signal_router = SignalRouter()
        
        # Initialize strategies
        await self._initialize_strategies()
        
        # Initialize notification system if enabled
        if self.config['notifications']['discord']['enabled']:
            self.notifier = DiscordNotifier(self.config['notifications']['discord'])
            await self.notifier.initialize()
        
        # Initialize UI if enabled
        if self.config['monitoring']['terminal_display']:
            self._initialize_display()
        
        logger.info("BotEngine initialization complete")

    async def _initialize_strategies(self) -> None:
        """Load and initialize all active trading strategies."""
        from strategies import STRATEGY_MAP  # Dynamic strategy imports
        
        active_strategies = self.config['strategies']['active_strategies']
        strategy_configs = self.config['strategies']['strategy_config']
        
        for strategy_name in active_strategies:
            if strategy_name in STRATEGY_MAP:
                strategy_class = STRATEGY_MAP[strategy_name]
                config = strategy_configs.get(strategy_name, {})
                
                strategy = strategy_class(config)
                await strategy.initialize(self.data_fetcher)
                self.strategies.append(strategy)
                
                logger.info(f"Initialized strategy: {strategy_name}")
            else:
                logger.warning(f"Strategy not found: {strategy_name}")

    def _initialize_display(self) -> None:
        """Initialize the Rich terminal display."""
        self.display_table = Table(title="Trading Bot Status", show_header=True)
        self.display_table.add_column("Metric")
        self.display_table.add_column("Value")
        
        # Use Live in manual start/stop mode so we can control lifecycle
        self.live_display = Live(self.display_table, refresh_per_second=4)
        try:
            self.live_display.start()
        except Exception:
            # If Live cannot start (e.g., not a TTY), continue without UI
            logger.debug("Rich Live display could not be started (non-interactive environment)")
            self.live_display = None

    async def run(self) -> None:
        """Main trading loop."""
        logger.info(f"Starting bot in {self.mode.name} mode")
        
        try:
            while self.state.running:
                if self.state.paused:
                    await asyncio.sleep(1)
                    continue
                
                # Main trading cycle
                await self._trading_cycle()
                
                # Update display
                if self.live_display:
                    self._update_display()
                
                # Sleep based on configured interval
                await asyncio.sleep(
                    self.config['monitoring']['update_interval']
                )
                
        except Exception as e:
            logger.error(f"Error in main trading loop: {str(e)}", exc_info=True)
            await self._emergency_shutdown()
            raise

    async def _trading_cycle(self) -> None:
        """Execute one complete trading cycle."""
        # 1. Fetch market data
        market_data = await self.data_fetcher.fetch_market_data()
        
        # 2. Generate signals from strategies
        signals = []
        for strategy in self.strategies:
            strategy_signals = await strategy.generate_signals(market_data)
            signals.extend(strategy_signals)
        
        # 3. Route signals through risk management
        approved_signals = []
        for signal in signals:
            if await self.risk_manager.evaluate_signal(signal, market_data):
                approved_signals.append(signal)
        
        # 4. Execute orders
        for signal in approved_signals:
            order_result = await self.order_manager.execute_order(signal)
            
            # Update performance metrics
            if order_result and 'pnl' in order_result:
                self._update_performance_metrics(order_result['pnl'])
                # Record equity progression after each trade execution
                try:
                    await self.record_trade_equity(order_result)
                except Exception:
                    logger.exception("Failed to record trade equity in trading cycle")
            
            # Send notifications
            if self.notifier:
                await self.notifier.send_order_notification(order_result)
        
        # 5. Update bot state
        await self._update_state()

    async def _update_state(self) -> None:
        """Update the bot's internal state."""
        self.state.balance = await self.order_manager.get_balance()
        self.state.equity = await self.order_manager.get_equity()
        self.state.active_orders = await self.order_manager.get_active_order_count()
        self.state.open_positions = await self.order_manager.get_open_position_count()

    def _update_performance_metrics(self, pnl: float) -> None:
        """Update performance tracking metrics."""
        try:
            # Guard initialization
            if 'total_pnl' not in self.performance_stats:
                self.performance_stats.setdefault('total_pnl', 0.0)
            if 'equity_history' not in self.performance_stats:
                self.performance_stats.setdefault('equity_history', [])
            if 'returns_history' not in self.performance_stats:
                self.performance_stats.setdefault('returns_history', [])
            if 'wins' not in self.performance_stats:
                self.performance_stats.setdefault('wins', 0)
            if 'losses' not in self.performance_stats:
                self.performance_stats.setdefault('losses', 0)
            if 'equity_progression' not in self.performance_stats:
                self.performance_stats.setdefault('equity_progression', [])
    
            # Update totals
            self.performance_stats['total_pnl'] += float(pnl)
    
            # Track returns for Sharpe ratio calculation
            current_equity = float(self.state.equity or 0.0)
            equity_history = self.performance_stats['equity_history']
            returns_history = self.performance_stats['returns_history']
    
            if equity_history:
                prev_equity = equity_history[-1]
                if prev_equity > 0:
                    daily_return = (current_equity - prev_equity) / prev_equity
                    returns_history.append(daily_return)
    
            equity_history.append(current_equity)
    
            # Calculate win/loss counts
            if pnl > 0:
                self.performance_stats['wins'] += 1
            elif pnl < 0:
                self.performance_stats['losses'] += 1
    
            total_trades = self.performance_stats['wins'] + self.performance_stats['losses']
            if total_trades > 0:
                self.performance_stats['win_rate'] = self.performance_stats['wins'] / total_trades
    
            # Calculate max drawdown
            if len(equity_history) > 1:
                peak = max(equity_history)
                trough = min(equity_history)
                if peak > 0:
                    max_dd = (peak - trough) / peak
                    self.performance_stats['max_drawdown'] = max(max_dd, float(self.performance_stats.get('max_drawdown', 0.0)))
    
            # Calculate Sharpe ratio (annualized)
            if len(returns_history) > 1 and np.std(returns_history) > 0:
                returns = np.array(returns_history)
                risk_free_rate = 0.0  # Can be configured
                excess_returns = returns - risk_free_rate
                sharpe = float(np.mean(excess_returns) / np.std(excess_returns))
                self.performance_stats['sharpe_ratio'] = sharpe * np.sqrt(252)  # Annualize
        except Exception:
            logger.exception("Failed to update performance metrics")

    async def record_trade_equity(self, order_result: Dict) -> None:
        """
        Record equity progression after a trade execution.

        Adds a dict to performance_stats['equity_progression'] with:
          - trade_id
          - timestamp
          - equity
          - pnl
          - cumulative_return (relative to starting balance)
        """
        try:
            if not order_result:
                return

            # Ensure equity_progression exists
            equity_prog = self.performance_stats.setdefault('equity_progression', [])

            # Get current equity from order manager (async)
            try:
                current_equity = await self.order_manager.get_equity()
            except Exception:
                # Fallback to state.equity if order_manager can't provide it
                current_equity = self.state.equity or 0.0

            # Normalize values
            trade_id = order_result.get('id', f"trade_{int(time.time()*1000)}")
            timestamp = order_result.get('timestamp', int(time.time()*1000))
            pnl = order_result.get('pnl', None)
            # coerce current_equity safely
            try:
                equity_val = float(current_equity) if current_equity is not None else 0.0
            except Exception:
                equity_val = 0.0

            # For backtest/paper modes the OrderManager may not track balance.
            # In that case derive equity from starting_balance + total_pnl so progression is meaningful.
            try:
                if getattr(self, "mode", None) in (TradingMode.BACKTEST, TradingMode.PAPER):
                    # If order_manager returned 0 or not tracking, compute from starting balance + total_pnl
                    if equity_val == 0.0:
                        equity_val = float(getattr(self, "starting_balance", 0.0)) + float(self.performance_stats.get('total_pnl', 0.0))
            except Exception:
                # Ignore and proceed with current equity_val
                pass

            # Calculate cumulative return relative to starting balance
            try:
                cumulative_return = 0.0
                if getattr(self, "starting_balance", None) and float(self.starting_balance) > 0:
                    cumulative_return = (equity_val - float(self.starting_balance)) / float(self.starting_balance)
            except Exception:
                cumulative_return = 0.0

            record = {
                'trade_id': trade_id,
                'timestamp': timestamp,
                'equity': equity_val,
                'pnl': pnl,
                'cumulative_return': cumulative_return
            }

            equity_prog.append(record)

        except Exception:
            logger.exception("Failed to record trade equity")

    def _update_display(self) -> None:
        """Update the Rich terminal display."""
        if not self.display_table:
            return
            
        self.display_table.rows = [
            ("Mode", self.mode.name),
            ("Status", "PAUSED" if self.state.paused else "RUNNING"),
            ("Balance", f"{self.state.balance:.2f} {self.config['exchange']['base_currency']}"),
            ("Equity", f"{self.state.equity:.2f} {self.config['exchange']['base_currency']}"),
            ("Active Orders", str(self.state.active_orders)),
            ("Open Positions", str(self.state.open_positions)),
            ("Total PnL", f"{self.performance_stats['total_pnl']:.2f}"),
            ("Win Rate", f"{self.performance_stats['win_rate']:.2%}"),
        ]

    async def shutdown(self) -> None:
        """Gracefully shutdown the bot engine."""
        logger.info("Shutting down BotEngine")
        self.state.running = False
        
        # Shutdown modules in reverse order
        if self.notifier:
            await self.notifier.shutdown()
        
        if self.order_manager:
            await self.order_manager.shutdown()
        
        if self.data_fetcher:
            await self.data_fetcher.shutdown()
        
        if self.live_display:
            try:
                self.live_display.stop()
            except Exception:
                logger.debug("Error stopping live display", exc_info=True)
            self.live_display = None
        
        logger.info("BotEngine shutdown complete")

    async def _emergency_shutdown(self) -> None:
        """Execute emergency shutdown procedures."""
        logger.critical("Executing emergency shutdown!")
        
        if self.notifier:
            await self.notifier.send_alert("🚨 EMERGENCY SHUTDOWN TRIGGERED!")
        
        # Cancel all open orders
        if self.order_manager:
            await self.order_manager.cancel_all_orders()
        
        await self.shutdown()
