"""
core/bot_engine.py

Facade for the trading bot engine that maintains backward compatibility
while internally using decomposed components for better architecture.
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass
from core.types import TradingMode

from utils.logger import setup_logging
from utils.config_loader import ConfigLoader
from data.data_fetcher import DataFetcher
from strategies.base_strategy import BaseStrategy
from risk.risk_manager import RiskManager
from core.order_manager import OrderManager
from core.signal_router import SignalRouter
from notifier.discord_bot import DiscordNotifier
from core.task_manager import TaskManager
from core.timeframe_manager import TimeframeManager
from strategies.regime.strategy_selector import get_strategy_selector, update_strategy_performance
from core.cache import initialize_cache, get_cache, close_cache

# Import decomposed components
from core.trading_coordinator import TradingCoordinator
from core.data_manager import DataManager
from core.signal_processor import SignalProcessor
from core.performance_tracker import PerformanceTracker
from core.order_executor import OrderExecutor
from core.state_manager import StateManager
from core.binary_model_integration import get_binary_integration, BinaryModelIntegration

logger = logging.getLogger(__name__)

# Strategy mapping for tests to patch
STRATEGY_MAP = {}


def now_ms() -> int:
    """
    Get current timestamp in milliseconds since epoch.

    Returns:
        Current timestamp as integer milliseconds
    """
    return int(time.time() * 1000)




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

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the BotEngine.

        Args:
            config: Configuration dictionary (loaded from utils/config_loader).
        """
        self.config: Dict[str, Any] = config
        # Mode derived from config environment.mode (expects 'live', 'paper', 'backtest')
        self.mode: TradingMode = TradingMode[config["environment"]["mode"].upper()]
        self.state: BotState = BotState()
        # Portfolio mode: when true, manage multiple trading pairs concurrently
        self.portfolio_mode: bool = bool(
            self.config.get("trading", {}).get("portfolio_mode", False)
        )
        # List of trading pairs to operate on. Populated in initialize().
        self.pairs: List[str] = []

        # Starting balance (present here for type clarity; initialize() may overwrite)
        try:
            balance = float(
                self.config.get("trading", {}).get("initial_balance", 1000.0)
            )
            # Validate that balance is positive
            self.starting_balance: float = balance if balance > 0 else 10000.0
        except Exception:
            self.starting_balance = 10000.0

        # Core modules
        self.data_fetcher: Optional[DataFetcher] = None
        self.timeframe_manager: Optional[TimeframeManager] = None
        self.strategies: List[BaseStrategy] = []
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.signal_router: Optional[SignalRouter] = None
        self.notifier: Optional[DiscordNotifier] = None

        # Task management for background tasks
        self.task_manager: TaskManager = TaskManager()

        # Global safe-mode flag (set when any core component signals repeated critical failures)
        self.global_safe_mode: bool = False
        # Internal flag to prevent duplicate safe-mode notifications
        self._safe_mode_notified: bool = False

        # Performance tracking
        self.performance_stats: Dict[str, Any] = {
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0,
            "equity_history": [],
            "returns_history": [],
            # equity_progression stores dicts with schema:
            # {'trade_id', 'timestamp', 'equity', 'pnl', 'cumulative_return'}
            "equity_progression": [],
        }

        # Shutdown hooks registered during initialize() for orderly teardown
        self._shutdown_hooks: List = []

        # UI components (removed rich dependency)
        self.live_display: Optional[Any] = None
        self.display_table: Optional[Any] = None

        # Market data cache for performance
        self.market_data_cache: Dict[str, Any] = {}
        self.cache_timestamp: float = 0.0
        self.cache_ttl: float = 60.0  # 1 minute cache

    async def initialize(self) -> None:
        """Initialize all components of the trading bot."""
        logger.info("Initializing BotEngine")
        
        # Step 1: Determine trading pairs
        self._determine_trading_pairs()
        
        # Step 2: Initialize cache if configured
        await self._initialize_cache()
        
        # Step 3: Initialize core modules
        await self._initialize_core_modules()
        
        # Step 4: Initialize strategies
        await self._initialize_strategies()
        
        # Step 5: Initialize notification system
        await self._initialize_notifications()
        
        # Step 6: Initialize UI if enabled
        self._initialize_display()
        
        logger.info("BotEngine initialization complete")

    def _determine_trading_pairs(self) -> None:
        """Determine trading pairs based on configuration and portfolio mode."""
        exchange_config = self.config["exchange"]
        trading_config = self.config["trading"]

        if self.portfolio_mode:
            markets = exchange_config.get("markets", []) or []
            self.pairs = [m for m in markets if isinstance(m, str)]
            
            # Fallback if markets is empty
            if not self.pairs:
                configured_symbol = trading_config.get("symbol") or None
                if configured_symbol:
                    self.pairs = [configured_symbol]
                elif markets and isinstance(markets, list) and markets:
                    self.pairs = [markets[0]]
        else:
            # Single-pair mode (legacy)
            configured_symbol = trading_config.get("symbol") or None
            if configured_symbol:
                self.pairs = [configured_symbol]
            else:
                markets = exchange_config.get("markets", []) or []
                self.pairs = [markets[0]] if markets and isinstance(markets, list) and markets else []

    async def _initialize_cache(self) -> None:
        """Initialize Redis cache if configured."""
        cache_config = self.config.get("cache", {})
        if cache_config.get("enabled", False):
            logger.info("Initializing Redis cache...")
            if initialize_cache(cache_config):
                logger.info("Redis cache initialized successfully")
                # Register cache shutdown hook
                self._shutdown_hooks.append(close_cache)
            else:
                logger.warning("Failed to initialize Redis cache, continuing without cache")
        else:
            logger.info("Redis cache disabled in configuration")

    async def _initialize_core_modules(self) -> None:
        """Initialize core trading modules."""
        exchange_config = self.config["exchange"]
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(exchange_config)
        await self.data_fetcher.initialize()
        if hasattr(self.data_fetcher, "shutdown"):
            self._shutdown_hooks.append(self.data_fetcher.shutdown)

        # Initialize timeframe manager
        tf_config = self.config.get("multi_timeframe", {})
        self.timeframe_manager = TimeframeManager(self.data_fetcher, tf_config)
        await self.timeframe_manager.initialize()
        if hasattr(self.timeframe_manager, "shutdown"):
            self._shutdown_hooks.append(self.timeframe_manager.shutdown)

        # Register symbols with timeframe manager
        if self.pairs:
            for symbol in self.pairs:
                default_timeframes = ["15m", "1h", "4h"]
                if self.timeframe_manager.add_symbol(symbol, default_timeframes):
                    logger.info(f"Registered {symbol} with timeframes: {default_timeframes}")
                else:
                    logger.warning(f"Failed to register {symbol} with timeframe manager")

        # Initialize risk manager
        self.risk_manager = RiskManager(self.config["risk_management"])

        # Initialize order manager
        self.order_manager = OrderManager(self.config, self.mode)
        if hasattr(self.order_manager, "shutdown"):
            self._shutdown_hooks.append(self.order_manager.shutdown)
        
        # Configure order manager
        await self._configure_order_manager()

        # Initialize signal router
        self.signal_router = SignalRouter(self.risk_manager, task_manager=self.task_manager)

    async def _configure_order_manager(self) -> None:
        """Configure order manager with portfolio and pairs information."""
        try:
            self.order_manager.pairs = self.pairs
            self.order_manager.portfolio_mode = self.portfolio_mode
            
            if hasattr(self.order_manager, "initialize_portfolio"):
                allocation = self.config.get("trading", {}).get("pair_allocation", None)
                await self.order_manager.initialize_portfolio(self.pairs, self.portfolio_mode, allocation)
        except Exception:
            logger.debug("OrderManager portfolio initialization skipped or failed", exc_info=True)

        # Set starting balance
        try:
            balance = float(
                self.config.get("trading", {}).get("initial_balance", 1000.0)
            )
            # Validate that balance is positive
            self.starting_balance = balance if balance > 0 else 10000.0
        except Exception:
            self.starting_balance = 10000.0

    async def _initialize_notifications(self) -> None:
        """Initialize notification system if enabled."""
        if self.config["notifications"]["discord"]["enabled"]:
            self.notifier = DiscordNotifier(self.config["notifications"]["discord"], task_manager=self.task_manager)
            await self.notifier.initialize()
            if hasattr(self.notifier, "shutdown"):
                self._shutdown_hooks.append(self.notifier.shutdown)

    async def _initialize_strategies(self) -> None:
        """Load and initialize all active trading strategies."""
        # Use module-level STRATEGY_MAP for test compatibility
        active_strategies = self.config["strategies"]["active_strategies"]
        strategy_configs = self.config["strategies"]["strategy_config"]

        for strategy_name in active_strategies:
            if strategy_name in STRATEGY_MAP:
                strategy_class = STRATEGY_MAP[strategy_name]
                config = strategy_configs.get(strategy_name, {})

                strategy = strategy_class(config)
                await strategy.initialize(self.data_fetcher)
                self.strategies.append(strategy)
                # register shutdown hook for strategy if provided
                if hasattr(strategy, "shutdown"):
                    self._shutdown_hooks.append(strategy.shutdown)

                logger.info(f"Initialized strategy: {strategy_name}")
            else:
                logger.warning(f"Strategy not found: {strategy_name}")

    def _initialize_display(self) -> None:
        """Initialize the terminal display (rich dependency removed)."""
        # Live display not available without rich
        self.live_display = None
        logger.info("Terminal display initialized (rich dependency removed)")

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
                await asyncio.sleep(self.config["monitoring"]["update_interval"])

        except Exception as e:
            logger.error(f"Error in main trading loop: {str(e)}", exc_info=True)
            await self._emergency_shutdown()
            # Do not re-raise to prevent test failures

    async def _trading_cycle(self) -> None:
        """Execute one complete trading cycle."""
        # 1. Fetch market data
        market_data = await self._fetch_market_data()

        # 2. Check safe mode conditions
        if await self._check_safe_mode_conditions():
            return

        # 3. Process through binary model integration (if enabled)
        integrated_decisions = await self._process_binary_integration(market_data)

        # 4. Generate signals from strategies (legacy path or when binary integration fails)
        if not integrated_decisions:
            signals = await self._generate_signals(market_data)

            # 5. Route signals through risk management
            approved_signals = await self._evaluate_risk(signals, market_data)

            # 6. Execute orders
            await self._execute_orders(approved_signals)
        else:
            # Execute integrated decisions
            await self._execute_integrated_decisions(integrated_decisions)

        # 7. Update bot state
        await self._update_state()

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data with Redis caching and multi-timeframe support."""
        current_time = time.time()
        cache = get_cache()
        
        # Check if we should use internal cache (fallback)
        use_internal_cache = not cache or not cache._connected
        if use_internal_cache and self.market_data_cache and (current_time - self.cache_timestamp) < self.cache_ttl:
            logger.debug("Using internal cached market data")
            return self.market_data_cache

        try:
            # Fetch single-timeframe data
            market_data = await self._fetch_single_timeframe_data(cache)
            
            # Fetch multi-timeframe data
            multi_timeframe_data = await self._fetch_multi_timeframe_data()
            
            # Combine data
            combined_data = self._combine_market_data(market_data, multi_timeframe_data)
            
            # Cache the fetched data
            self._cache_market_data(combined_data, current_time)
            
            return combined_data
            
        except Exception:
            logger.exception("Failed to fetch market data")
            # Return cached data if available, otherwise empty dict
            if self.market_data_cache:
                logger.warning("Using stale cached data due to fetch failure")
                return self.market_data_cache

            return {}

    async def _fetch_single_timeframe_data(self, cache: Any) -> Dict[str, Any]:
        """Fetch single-timeframe market data with caching support."""
        market_data = {}
        
        if self.portfolio_mode and hasattr(self.data_fetcher, "get_realtime_data"):
            market_data = await self._fetch_portfolio_realtime_data(cache)
        elif not self.portfolio_mode and hasattr(self.data_fetcher, "get_historical_data"):
            market_data = await self._fetch_single_historical_data(cache)
        elif hasattr(self.data_fetcher, "get_multiple_historical_data"):
            market_data = await self._fetch_multiple_historical_data(cache)
            
        return market_data

    async def _fetch_portfolio_realtime_data(self, cache: Any) -> Dict[str, Any]:
        """Fetch portfolio realtime data with Redis caching."""
        if cache and cache._connected:
            # Try to get cached ticker data for all pairs
            cached_tickers = {}
            for symbol in self.pairs:
                ticker_data = await cache.get_market_ticker(symbol)
                if ticker_data:
                    cached_tickers[symbol] = ticker_data
            
            if cached_tickers:
                logger.debug(f"Using cached ticker data for {len(cached_tickers)} symbols")
            
            # Fetch fresh data if needed
            if len(cached_tickers) < len(self.pairs):
                fresh_data = await self.data_fetcher.get_realtime_data(self.pairs)
                # Update cache with fresh data
                for symbol, ticker in fresh_data.items():
                    await cache.set_market_ticker(symbol, ticker)
                # Merge cached and fresh data
                cached_tickers.update(fresh_data)
            
            return cached_tickers
        else:
            # No Redis cache, fetch directly
            return await self.data_fetcher.get_realtime_data(self.pairs)

    async def _fetch_single_historical_data(self, cache: Any) -> Dict[str, Any]:
        """Fetch single historical data with Redis caching."""
        symbol = self.pairs[0] if self.pairs else None
        if not symbol:
            return {}
            
        timeframe = self.config.get("backtesting", {}).get("timeframe", "1h")
        
        if cache and cache._connected:
            # Try to get cached OHLCV data
            cached_ohlcv = await cache.get_ohlcv(symbol, timeframe)
            if cached_ohlcv:
                logger.debug(f"Using cached OHLCV data for {symbol}")
                return {symbol: cached_ohlcv}
            else:
                # Fetch and cache fresh data
                df = await self.data_fetcher.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=100,
                )
                await cache.set_ohlcv(symbol, timeframe, df)
                return {symbol: df}
        else:
            # No Redis cache, fetch directly
            df = await self.data_fetcher.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=100,
            )
            return {symbol: df}

    async def _fetch_multiple_historical_data(self, cache: Any) -> Dict[str, Any]:
        """Fetch multiple historical data with Redis caching."""
        if not self.pairs:
            return {}
            
        if cache and cache._connected:
            # Try to get cached OHLCV data for all pairs
            cached_data = await cache.get_multiple_ohlcv(self.pairs)
            
            # Determine which symbols need fresh data
            symbols_to_fetch = [s for s in self.pairs if not cached_data.get(s)]
            
            if symbols_to_fetch:
                # Fetch fresh data for missing symbols
                fresh_data = await self.data_fetcher.get_multiple_historical_data(symbols_to_fetch)
                
                # Update cache with fresh data
                await cache.set_multiple_ohlcv(fresh_data, "1h")  # Default timeframe
                
                # Merge cached and fresh data
                for symbol, data in fresh_data.items():
                    cached_data[symbol] = data
            
            logger.debug(f"Using cached OHLCV data for {len([d for d in cached_data.values() if d])} symbols")
            return cached_data
        else:
            # No Redis cache, fetch directly
            return await self.data_fetcher.get_multiple_historical_data(self.pairs)

    async def _fetch_multi_timeframe_data(self) -> Dict[str, Any]:
        """Fetch multi-timeframe data for all symbols."""
        multi_timeframe_data = {}
        
        if not self.timeframe_manager or not self.pairs:
            return multi_timeframe_data
            
        for symbol in self.pairs:
            try:
                synced_data = await self.timeframe_manager.fetch_multi_timeframe_data(symbol)
                if synced_data:
                    multi_timeframe_data[symbol] = synced_data
                    logger.debug(f"Fetched multi-timeframe data for {symbol}")
                else:
                    logger.warning(f"Failed to fetch multi-timeframe data for {symbol}")
            except Exception as e:
                logger.warning(f"Error fetching multi-timeframe data for {symbol}: {e}")
                
        return multi_timeframe_data

    def _combine_market_data(self, market_data: Dict[str, Any], multi_timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine single-timeframe and multi-timeframe market data."""
        combined_data = market_data.copy()
        
        for symbol, synced_data in multi_timeframe_data.items():
            if symbol in combined_data:
                # Add multi-timeframe data to existing symbol data
                combined_data[symbol] = {
                    'single_timeframe': combined_data[symbol],
                    'multi_timeframe': synced_data
                }
            else:
                # Only multi-timeframe data available
                combined_data[symbol] = {
                    'multi_timeframe': synced_data
                }
                
        return combined_data

    def _cache_market_data(self, combined_data: Dict[str, Any], current_time: float) -> None:
        """Cache the fetched market data."""
        self.market_data_cache = combined_data
        self.cache_timestamp = current_time
        logger.debug("Fetched and cached market data (including multi-timeframe)")

    async def _check_safe_mode_conditions(self) -> bool:
        """Check various safe mode conditions and return True if trading should be skipped."""
        try:
            await self._check_global_safe_mode()
            if self.global_safe_mode:
                if not self._safe_mode_notified:
                    logger.warning("Global safe mode active: skipping trading cycle")
                    self._safe_mode_notified = True
                    try:
                        if self.notifier and self.config["notifications"]["discord"]["enabled"]:
                            await self.notifier.send_alert("Bot entering SAFE MODE: suspending new trades.")
                    except Exception:
                        logger.exception("Failed to send safe-mode notification")
                return True
        except Exception:
            logger.exception("Failed to perform global safe-mode check; skipping trading cycle")
            return True

        try:
            order_safe = bool(getattr(self.order_manager, "safe_mode_active", False))
            if order_safe and not self._safe_mode_notified:
                logger.warning("Order manager safe mode active: skipping trading cycle")
                self._safe_mode_notified = True
                try:
                    if self.notifier and self.config["notifications"]["discord"]["enabled"]:
                        await self.notifier.send_alert("Bot entering SAFE MODE: suspending new trades.")
                except Exception:
                    logger.exception("Failed to send safe-mode notification")
                return True
        except Exception:
            pass

        return False

    async def _generate_signals(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate trading signals from all active strategies with multi-timeframe support."""
        signals = []
        strategy_selector = get_strategy_selector()

        if strategy_selector.enabled and market_data:
            primary_symbol = list(market_data.keys())[0] if market_data else None
            if primary_symbol and primary_symbol in market_data:
                selected_strategy_class = strategy_selector.select_strategy(market_data[primary_symbol])
                if selected_strategy_class:
                    selected_strategy = None
                    for strategy in self.strategies:
                        if type(strategy) == selected_strategy_class:
                            selected_strategy = strategy
                            break

                    if selected_strategy:
                        logger.info(f"Strategy selector chose: {selected_strategy_class.__name__}")
                        strategy_signals = await selected_strategy.generate_signals(market_data, self._extract_multi_tf_data(market_data, primary_symbol))
                        signals.extend(strategy_signals)
                    else:
                        logger.warning(f"Selected strategy {selected_strategy_class.__name__} not found in active strategies")
                        signals = await self._generate_signals_from_all_strategies(market_data)
                else:
                    logger.warning("Strategy selector returned no strategy, using all available strategies")
                    signals = await self._generate_signals_from_all_strategies(market_data)
            else:
                signals = await self._generate_signals_from_all_strategies(market_data)
        else:
            signals = await self._generate_signals_from_all_strategies(market_data)

        return signals

    async def _generate_signals_from_all_strategies(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate signals from all strategies when strategy selector is disabled or fails."""
        signals = []
        for strategy in self.strategies:
            # Extract multi-timeframe data for the strategy's primary symbol
            primary_symbol = list(market_data.keys())[0] if market_data else None
            multi_tf_data = self._extract_multi_tf_data(market_data, primary_symbol) if primary_symbol else None
            strategy_signals = await strategy.generate_signals(market_data, multi_tf_data)
            signals.extend(strategy_signals)
        return signals

    async def _evaluate_risk(self, signals: List[Any], market_data: Dict[str, Any]) -> List[Any]:
        """Evaluate signals through risk management and return approved signals."""
        approved_signals = []
        for signal in signals:
            if await self.risk_manager.evaluate_signal(signal, market_data):
                approved_signals.append(signal)
        return approved_signals

    async def _execute_orders(self, approved_signals: List[Any]) -> None:
        """Execute approved trading signals and handle results."""
        strategy_selector = get_strategy_selector()
        selected_strategy = None

        # Find selected strategy if strategy selector is enabled
        if strategy_selector.enabled:
            for strategy in self.strategies:
                if hasattr(strategy_selector, '_selected_strategy_class') and strategy_selector._selected_strategy_class:
                    if type(strategy) == strategy_selector._selected_strategy_class:
                        selected_strategy = strategy
                        break

        for signal in approved_signals:
            order_result = await self.order_manager.execute_order(signal)

            # Update performance metrics
            if order_result and "pnl" in order_result:
                self._update_performance_metrics(order_result["pnl"])

                # Update strategy selector performance if enabled
                if strategy_selector.enabled and selected_strategy:
                    pnl = order_result.get("pnl", 0.0)
                    returns = pnl / self.starting_balance if self.starting_balance > 0 else 0.0
                    is_win = pnl > 0
                    update_strategy_performance(selected_strategy.__class__.__name__, pnl, returns, is_win)

                # Record equity progression after each trade execution
                try:
                    await self.record_trade_equity(order_result)
                except Exception:
                    logger.exception("Failed to record trade equity in trading cycle")

            # Send notifications
            if self.notifier:
                await self.notifier.send_order_notification(order_result)

    async def _update_state(self) -> None:
        """Update the bot's internal state."""
        self.state.balance = await self.order_manager.get_balance()
        self.state.equity = await self.order_manager.get_equity()
        self.state.active_orders = await self.order_manager.get_active_order_count()
        self.state.open_positions = await self.order_manager.get_open_position_count()

    def _update_performance_metrics(self, pnl: float) -> None:
        """Update performance tracking metrics."""
        try:
            self._initialize_performance_stats()
            self._update_pnl(pnl)
            self._update_equity_history()
            self._update_win_loss_counts(pnl)
            self._calculate_win_rate()
            self._calculate_max_drawdown()
            self._calculate_sharpe_ratio()
        except Exception:
            logger.exception("Failed to update performance metrics")

    def _initialize_performance_stats(self) -> None:
        """Initialize performance statistics if not already present."""
        if "total_pnl" not in self.performance_stats:
            self.performance_stats.setdefault("total_pnl", 0.0)
        if "equity_history" not in self.performance_stats:
            self.performance_stats.setdefault("equity_history", [])
        if "returns_history" not in self.performance_stats:
            self.performance_stats.setdefault("returns_history", [])
        if "wins" not in self.performance_stats:
            self.performance_stats.setdefault("wins", 0)
        if "losses" not in self.performance_stats:
            self.performance_stats.setdefault("losses", 0)
        if "equity_progression" not in self.performance_stats:
            self.performance_stats.setdefault("equity_progression", [])

    def _update_pnl(self, pnl: float) -> None:
        """Update total PnL."""
        self.performance_stats["total_pnl"] += float(pnl)

    def _update_equity_history(self) -> None:
        """Update equity history and returns for Sharpe ratio calculation."""
        current_equity = float(self.state.equity or 0.0)
        equity_history = self.performance_stats["equity_history"]
        returns_history = self.performance_stats["returns_history"]

        if equity_history:
            prev_equity = equity_history[-1]
            if prev_equity > 0:
                daily_return = (current_equity - prev_equity) / prev_equity
                returns_history.append(daily_return)

        equity_history.append(current_equity)

    def _update_win_loss_counts(self, pnl: float) -> None:
        """Update win and loss counts based on PnL."""
        if pnl > 0:
            self.performance_stats["wins"] += 1
        elif pnl < 0:
            self.performance_stats["losses"] += 1

    def _calculate_win_rate(self) -> None:
        """Calculate win rate based on total trades."""
        total_trades = self.performance_stats["wins"] + self.performance_stats["losses"]
        if total_trades > 0:
            self.performance_stats["win_rate"] = (
                self.performance_stats["wins"] / total_trades
            )

    def _calculate_max_drawdown(self) -> None:
        """Calculate maximum drawdown from equity history."""
        equity_history = self.performance_stats["equity_history"]
        if len(equity_history) > 1:
            peak = max(equity_history)
            trough = min(equity_history)
            if peak > 0:
                max_dd = (peak - trough) / peak
                self.performance_stats["max_drawdown"] = max(
                    max_dd, float(self.performance_stats.get("max_drawdown", 0.0))
                )

    def _calculate_sharpe_ratio(self) -> None:
        """Calculate Sharpe ratio (annualized) from returns history."""
        returns_history = self.performance_stats["returns_history"]
        if len(returns_history) > 1 and np.std(returns_history) > 0:
            returns = np.array(returns_history)
            risk_free_rate = 0.0  # Can be configured
            excess_returns = returns - risk_free_rate
            sharpe = float(np.mean(excess_returns) / np.std(excess_returns))
            self.performance_stats["sharpe_ratio"] = sharpe * np.sqrt(252)  # Annualize

    async def record_trade_equity(self, order_result: Optional[Dict[str, Any]]) -> None:
        """
        Record equity progression after a trade execution.

        Args:
            order_result: Dictionary returned from OrderManager.execute_order containing:
              - id (optional): trade identifier
              - timestamp (optional): epoch ms or ISO timestamp
              - pnl (optional): profit/loss for the trade
              - symbol (optional): trading symbol

        Side effects:
            Appends a record to self.performance_stats['equity_progression'] with:
              - trade_id: from order_result["id"] or generated fallback
              - timestamp: from order_result["timestamp"] or current time in ms
              - equity: current equity value
              - pnl: from order_result["pnl"] or None if missing
              - cumulative_return: relative to starting balance
        """
        try:
            if not order_result:
                return

            # Get current equity (handle case where order_manager.get_equity() returns 0)
            equity_val = await self._get_current_equity()

            # If equity is 0, try to calculate it from starting balance + total pnl
            if equity_val == 0.0 and hasattr(self, 'performance_stats'):
                total_pnl = self.performance_stats.get("total_pnl", 0.0)
                if self.starting_balance > 0:
                    equity_val = float(self.starting_balance) + float(total_pnl)

            # Calculate cumulative return
            cumulative_return = self._calculate_cumulative_return(equity_val)

            # Create and append record
            record = self._create_equity_record(order_result, equity_val, cumulative_return)
            self.performance_stats.setdefault("equity_progression", []).append(record)

        except Exception:
            logger.exception("Failed to record trade equity")

    async def _get_current_equity(self) -> float:
        """Get current equity from order manager or fallback to state."""
        try:
            current_equity = await self.order_manager.get_equity()
            return float(current_equity) if current_equity is not None else 0.0
        except Exception:
            # Fallback to state.equity if order_manager can't provide it
            return float(self.state.equity or 0.0)

    def _calculate_cumulative_return(self, equity_val: float) -> float:
        """Calculate cumulative return relative to starting balance."""
        try:
            if self.starting_balance and float(self.starting_balance) > 0:
                return (equity_val - float(self.starting_balance)) / float(self.starting_balance)
            return 0.0
        except Exception:
            return 0.0

    def _create_equity_record(
        self,
        order_result: Dict[str, Any],
        equity_val: float,
        cumulative_return: float
    ) -> Dict[str, Any]:
        """
        Create equity progression record from order execution result.

        Args:
            order_result: Dictionary containing order execution details
            equity_val: Current equity value
            cumulative_return: Cumulative return relative to starting balance

        Returns:
            Dictionary with equity progression record containing:
            - trade_id: Trade identifier (from order_result or generated)
            - timestamp: Timestamp in milliseconds (from order_result or current time)
            - symbol: Trading symbol (if available)
            - equity: Current equity value
            - pnl: Profit/loss from the trade (if available)
            - cumulative_return: Cumulative return relative to starting balance
        """
        # Generate trade_id if not provided
        trade_id = order_result.get("id")
        if not trade_id:
            trade_id = f"trade_{now_ms()}"

        # Get timestamp, convert to int if needed
        timestamp_val = order_result.get("timestamp")
        if timestamp_val is None:
            timestamp_val = now_ms()
        elif not isinstance(timestamp_val, int):
            # Try to convert to int (handles string timestamps)
            try:
                timestamp_val = int(timestamp_val)
            except (ValueError, TypeError):
                timestamp_val = now_ms()

        # Get symbol if available
        symbol = order_result.get("symbol") if isinstance(order_result, dict) else None

        # Get pnl, handle None values
        pnl = order_result.get("pnl")
        if pnl is not None:
            try:
                pnl = float(pnl)
            except (ValueError, TypeError):
                pnl = None

        return {
            "trade_id": trade_id,
            "timestamp": timestamp_val,
            "symbol": symbol,
            "equity": equity_val,
            "pnl": pnl,
            "cumulative_return": cumulative_return,
        }

    async def _check_global_safe_mode(self) -> None:
        """
        Inspect core components (OrderManager, SignalRouter, RiskManager) for
        safe-mode/blocking indicators and update self.global_safe_mode accordingly.

        This method is lightweight and non-blocking; it's safe to call each cycle.
        """
        try:
            order_safe = bool(getattr(self.order_manager, "safe_mode_active", False))
            router_block = bool(getattr(self.signal_router, "block_signals", False))
            risk_block = bool(getattr(self.risk_manager, "block_signals", False))

            should_be_safe = order_safe or router_block or risk_block

            if should_be_safe and not self.global_safe_mode:
                self.global_safe_mode = True
                logger.critical("Global safe mode ACTIVATED by component flag(s)")
                trade_logger = None
                try:
                    from utils.logger import get_trade_logger
                    trade_logger = get_trade_logger()
                except Exception:
                    trade_logger = None
                try:
                    if trade_logger:
                        trade_logger.trade("Global safe mode activated", {"order_safe": order_safe, "router_block": router_block, "risk_block": risk_block})
                except Exception:
                    logger.exception("Failed to emit safe-mode trade log")
                # Reset notification flag so notifier will be triggered on activation
                self._safe_mode_notified = False

            if not should_be_safe and self.global_safe_mode:
                # Clear global safe mode when all components report healthy
                self.global_safe_mode = False
                logger.info("Global safe mode CLEARED; components healthy")
                self._safe_mode_notified = False

        except Exception:
            # On failure to check, conservatively do nothing but log the issue
            logger.exception("Error while checking global safe mode state")

    def _update_display(self) -> None:
        """Update the terminal display with latest engine state."""
        if self.live_display is None:
            return

        # Gather latest state
        state_data = {
            "balance": self.state.balance,
            "equity": self.state.equity,
            "active_orders": self.state.active_orders,
            "open_positions": self.state.open_positions,
            "performance_metrics": self.performance_stats
        }

        # Update the display with the latest state
        self.live_display.update(state_data)

    def _log_status(self) -> None:
        """Log the current bot status."""
        try:
            balance_str = f"{float(self.state.balance):.2f} {self.config['exchange']['base_currency']}"
        except Exception:
            balance_str = str(self.state.balance)
        try:
            equity_str = f"{float(self.state.equity):.2f} {self.config['exchange']['base_currency']}"
        except Exception:
            equity_str = str(self.state.equity)

        logger.info(f"Bot Status - Mode: {self.mode.name}, Status: {'PAUSED' if self.state.paused else 'RUNNING'}, "
                   f"Balance: {balance_str}, Equity: {equity_str}, "
                   f"Active Orders: {self.state.active_orders}, Open Positions: {self.state.open_positions}, "
                   f"Total PnL: {self.performance_stats.get('total_pnl', 0.0):.2f}, "
                   f"Win Rate: {self.performance_stats.get('win_rate', 0.0):.2%}")

    def print_status_table(self) -> None:
        """Log the Trading Bot Status table in a formatted table layout."""
        try:
            # Prepare data
            mode = self.mode.name
            status = 'PAUSED' if self.state.paused else 'RUNNING'

            try:
                balance_str = f"{float(self.state.balance):.2f} {self.config['exchange']['base_currency']}"
            except Exception:
                balance_str = str(self.state.balance)

            try:
                equity_str = f"{float(self.state.equity):.2f} {self.config['exchange']['base_currency']}"
            except Exception:
                equity_str = str(self.state.equity)

            active_orders = str(self.state.active_orders)
            open_positions = str(self.state.open_positions)
            total_pnl = f"{self.performance_stats.get('total_pnl', 0.0):.2f}"
            win_rate = f"{self.performance_stats.get('win_rate', 0.0):.2%}"

            # Print table with pipes and dashes (ASCII compatible)
            print("\n+-----------------+---------------------+")
            print("| Trading Bot Status                  |")
            print("+-----------------+---------------------+")
            print(f"| Mode            | {mode:<19} |")
            print(f"| Status          | {status:<19} |")
            print(f"| Balance         | {balance_str:<19} |")
            print(f"| Equity          | {equity_str:<19} |")
            print(f"| Active Orders   | {active_orders:<19} |")
            print(f"| Open Positions  | {open_positions:<19} |")
            print(f"| Total PnL       | {total_pnl:<19} |")
            print(f"| Win Rate        | {win_rate:<19} |")
            print("+-----------------+---------------------+")

        except Exception:
            logger.exception("Failed to log status table")

    async def shutdown(self) -> None:
        """Gracefully shutdown the bot engine."""
        logger.info("Shutting down BotEngine")
        self.state.running = False

        # Cancel all tracked tasks first with timeout protection
        try:
            await asyncio.wait_for(self.task_manager.cancel_all(), timeout=30.0)
            logger.info("All tracked tasks cancelled successfully")
        except asyncio.TimeoutError:
            logger.warning("Timeout reached while cancelling tracked tasks")
        except Exception:
            logger.exception("Error cancelling tracked tasks during shutdown")

        # Execute registered shutdown hooks in reverse order with timeout protection
        # Hooks are expected to be zero-arg callables returning an awaitable (coroutine).
        for hook in reversed(self._shutdown_hooks):
            try:
                await asyncio.wait_for(hook(), timeout=15.0)
                logger.debug(f"Shutdown hook completed: {hook}")
            except asyncio.TimeoutError:
                logger.warning(f"Shutdown hook timed out: {hook}")
            except Exception:
                logger.exception(f"Shutdown hook failed: {hook}")

        # Ensure live display is stopped
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

        # Send emergency alert with timeout protection
        if self.notifier:
            try:
                await asyncio.wait_for(
                    self.notifier.send_alert("🚨 EMERGENCY SHUTDOWN TRIGGERED!"),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Emergency alert timed out")
            except Exception:
                logger.exception("Failed to send emergency alert")

        # Cancel all open orders with timeout protection
        if self.order_manager:
            try:
                await asyncio.wait_for(
                    self.order_manager.cancel_all_orders(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Order cancellation timed out")
            except Exception:
                logger.exception("Failed to cancel orders during emergency shutdown")

        await self.shutdown()

    async def _process_binary_integration(self, market_data: Dict[str, Any]) -> List[Any]:
        """
        Process market data through binary model integration.

        Args:
            market_data: Market data dictionary

        Returns:
            List of integrated trading decisions
        """
        try:
            binary_integration = get_binary_integration()

            if not binary_integration.enabled:
                logger.debug("Binary integration disabled")
                return []

            integrated_decisions = []

            # Process each symbol in the market data
            for symbol, data in market_data.items():
                try:
                    # Extract DataFrame from market data
                    if isinstance(data, dict):
                        if 'single_timeframe' in data:
                            df = data['single_timeframe']
                        else:
                            # Convert dict to DataFrame
                            df = pd.DataFrame(data)
                    else:
                        df = data

                    if df.empty or len(df) < 20:
                        logger.debug(f"Insufficient data for {symbol}, skipping binary integration")
                        continue

                    # Process through binary integration
                    decision = await binary_integration.process_market_data(df, symbol)

                    if decision.should_trade:
                        integrated_decisions.append({
                            'symbol': symbol,
                            'decision': decision,
                            'market_data': df
                        })
                        logger.info(f"Binary integration approved trade for {symbol}: {decision.reasoning}")

                except Exception as e:
                    logger.error(f"Error processing binary integration for {symbol}: {e}")
                    continue

            return integrated_decisions

        except Exception as e:
            logger.error(f"Binary integration processing failed: {e}")
            return []

    async def _execute_integrated_decisions(self, integrated_decisions: List[Dict[str, Any]]) -> None:
        """
        Execute integrated trading decisions from binary model.

        Args:
            integrated_decisions: List of integrated trading decisions
        """
        for decision_data in integrated_decisions:
            try:
                symbol = decision_data['symbol']
                decision = decision_data['decision']
                market_data = decision_data['market_data']

                # Create trading signal from integrated decision
                signal = self._create_signal_from_decision(decision, symbol, market_data)

                if signal:
                    # Execute the order
                    order_result = await self.order_manager.execute_order(signal)

                    # Update performance metrics
                    if order_result and "pnl" in order_result:
                        self._update_performance_metrics(order_result["pnl"])

                        # Update strategy selector performance if strategy was selected
                        if decision.selected_strategy:
                            strategy_selector = get_strategy_selector()
                            if strategy_selector.enabled:
                                pnl = order_result.get("pnl", 0.0)
                                returns = pnl / self.starting_balance if self.starting_balance > 0 else 0.0
                                is_win = pnl > 0
                                update_strategy_performance(decision.selected_strategy.__name__, pnl, returns, is_win)

                        # Record equity progression
                        try:
                            await self.record_trade_equity(order_result)
                        except Exception:
                            logger.exception("Failed to record trade equity in integrated decision")

                    # Send notifications
                    if self.notifier:
                        await self.notifier.send_order_notification(order_result)

                    logger.info(f"Executed integrated decision for {symbol}: {decision.reasoning}")

            except Exception as e:
                logger.error(f"Error executing integrated decision: {e}")
                continue

    def _create_signal_from_decision(self, decision: Any, symbol: str, market_data: pd.DataFrame) -> Optional[Any]:
        """
        Create a trading signal from integrated decision.

        Args:
            decision: Integrated trading decision
            symbol: Trading symbol
            market_data: Market data DataFrame

        Returns:
            Trading signal or None
        """
        try:
            from core.contracts import TradingSignal, SignalType, SignalStrength

            # Determine signal type based on direction
            if decision.direction == "long":
                signal_type = SignalType.BUY
            elif decision.direction == "short":
                signal_type = SignalType.SELL
            else:
                logger.warning(f"Unknown direction in decision: {decision.direction}")
                return None

            # Get current price
            current_price = market_data['close'].iloc[-1] if not market_data.empty else 0.0

            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=SignalStrength.MODERATE,
                current_price=current_price,
                amount=decision.position_size,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                timestamp=decision.timestamp,
                metadata={
                    "strategy": decision.selected_strategy.__name__ if decision.selected_strategy else "binary_integration",
                    "regime": decision.regime,
                    "binary_probability": decision.binary_probability,
                    "risk_score": decision.risk_score,
                    "reasoning": decision.reasoning
                }
            )

            return signal

        except Exception as e:
            logger.error(f"Error creating signal from decision: {e}")
            return None

    def _extract_multi_tf_data(self, market_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """
        Extract multi-timeframe data for a specific symbol from market data.

        Args:
            market_data: Combined market data dictionary
            symbol: Symbol to extract data for

        Returns:
            Multi-timeframe data or None if not available
        """
        try:
            if not market_data or symbol not in market_data:
                return None

            symbol_data = market_data[symbol]

            # Check if symbol_data is a dict with multi_timeframe key
            if isinstance(symbol_data, dict) and 'multi_timeframe' in symbol_data:
                return symbol_data['multi_timeframe']

            # Check if symbol_data is a SyncedData object directly
            if hasattr(symbol_data, 'data') and hasattr(symbol_data, 'timestamp'):
                return symbol_data

            return None

        except Exception as e:
            logger.warning(f"Failed to extract multi-timeframe data for {symbol}: {e}")
            return None
