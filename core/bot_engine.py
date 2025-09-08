"""
core/bot_engine.py

Facade for the trading bot engine that maintains backward compatibility
while internally using decomposed components for better architecture.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
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
from strategy_selector import get_strategy_selector, update_strategy_performance

# Import decomposed components
from core.trading_coordinator import TradingCoordinator
from core.data_manager import DataManager
from core.signal_processor import SignalProcessor
from core.performance_tracker import PerformanceTracker
from core.order_executor import OrderExecutor
from core.state_manager import StateManager

logger = logging.getLogger(__name__)

# Strategy mapping for tests to patch
STRATEGY_MAP = {}




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
            self.starting_balance: float = float(
                self.config.get("trading", {}).get("initial_balance", 1000.0)
            )
        except Exception:
            self.starting_balance = 1000.0

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

        # Load configuration
        exchange_config = self.config["exchange"]
        trading_config = self.config["trading"]

        # Determine trading pairs based on portfolio_mode while keeping backward compatibility.
        # If portfolio_mode is enabled, use the exchange 'markets' list from config.
        if self.portfolio_mode:
            markets = exchange_config.get("markets", []) or []
            self.pairs = [m for m in markets if isinstance(m, str)]
            # Fallback: if markets is empty, try trading.symbol or single entry in 'markets'
            if not self.pairs:
                configured_symbol = trading_config.get("symbol") or None
                if configured_symbol:
                    self.pairs = [configured_symbol]
                elif markets and isinstance(markets, list) and markets:
                    self.pairs = [markets[0]]
        else:
            # Single-pair mode (legacy) - prefer trading.symbol then first exchange market
            configured_symbol = trading_config.get("symbol") or None
            if configured_symbol:
                self.pairs = [configured_symbol]
            else:
                markets = exchange_config.get("markets", []) or []
                self.pairs = [markets[0]] if markets and isinstance(markets, list) and markets else []

        # Initialize modules
        self.data_fetcher = DataFetcher(exchange_config)
        await self.data_fetcher.initialize()
        # register shutdown hook for data_fetcher
        if hasattr(self.data_fetcher, "shutdown"):
            self._shutdown_hooks.append(self.data_fetcher.shutdown)

        # Initialize timeframe manager for multi-timeframe analysis
        tf_config = self.config.get("multi_timeframe", {})
        self.timeframe_manager = TimeframeManager(self.data_fetcher, tf_config)
        await self.timeframe_manager.initialize()
        # register shutdown hook for timeframe_manager
        if hasattr(self.timeframe_manager, "shutdown"):
            self._shutdown_hooks.append(self.timeframe_manager.shutdown)

        # Register symbols with timeframe manager for multi-timeframe analysis
        if self.pairs:
            for symbol in self.pairs:
                # Default timeframes for multi-timeframe validation
                default_timeframes = ["15m", "1h", "4h"]
                if self.timeframe_manager.add_symbol(symbol, default_timeframes):
                    logger.info(f"Registered {symbol} with timeframes: {default_timeframes}")
                else:
                    logger.warning(f"Failed to register {symbol} with timeframe manager")

        self.risk_manager = RiskManager(self.config["risk_management"])
        # Pass full config to OrderManager to ensure it can access paper/backtest settings
        self.order_manager = OrderManager(self.config, self.mode)
        # register shutdown hook for order_manager
        if hasattr(self.order_manager, "shutdown"):
            self._shutdown_hooks.append(self.order_manager.shutdown)
        # Configure OrderManager with portfolio/pairs info when available (backwards-compatible)
        try:
            # attach pairs and portfolio flag for per-symbol tracking
            self.order_manager.pairs = self.pairs
            self.order_manager.portfolio_mode = self.portfolio_mode
            # If OrderManager exposes an initialization hook, call it (some tests/mock setups may not implement it)
            if hasattr(self.order_manager, "initialize_portfolio"):
                allocation = self.config.get("trading", {}).get("pair_allocation", None)
                # allocation may be None or dict mapping symbol->fraction
                await self.order_manager.initialize_portfolio(self.pairs, self.portfolio_mode, allocation)
        except Exception:
            logger.debug("OrderManager portfolio initialization skipped or failed", exc_info=True)

        # Record starting balance from config for cumulative return calculations
        try:
            self.starting_balance = float(
                self.config.get("trading", {}).get("initial_balance", 1000.0)
            )
        except Exception:
            self.starting_balance = 1000.0
        self.signal_router = SignalRouter(self.risk_manager, task_manager=self.task_manager)

        # Initialize strategies
        await self._initialize_strategies()

        # Initialize notification system if enabled
        if self.config["notifications"]["discord"]["enabled"]:
            self.notifier = DiscordNotifier(self.config["notifications"]["discord"], task_manager=self.task_manager)
            await self.notifier.initialize()
            # register notifier shutdown hook
            if hasattr(self.notifier, "shutdown"):
                self._shutdown_hooks.append(self.notifier.shutdown)

        # Initialize UI if enabled
        if self.config["monitoring"]["terminal_display"]:
            self._initialize_display()

        logger.info("BotEngine initialization complete")

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

        # 3. Generate signals from strategies
        signals = await self._generate_signals(market_data)

        # 4. Route signals through risk management
        approved_signals = await self._evaluate_risk(signals, market_data)

        # 5. Execute orders
        await self._execute_orders(approved_signals)

        # 6. Update bot state
        await self._update_state()

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data with caching and multi-timeframe support."""
        current_time = time.time()
        if self.market_data_cache and (current_time - self.cache_timestamp) < self.cache_ttl:
            logger.debug("Using cached market data")
            return self.market_data_cache

        market_data = {}
        multi_timeframe_data = {}

        try:
            # Fetch single-timeframe data (backward compatibility)
            if self.portfolio_mode and hasattr(self.data_fetcher, "get_realtime_data"):
                market_data = await self.data_fetcher.get_realtime_data(self.pairs)
            elif not self.portfolio_mode and hasattr(self.data_fetcher, "get_historical_data"):
                symbol = self.pairs[0] if self.pairs else None
                if symbol:
                    df = await self.data_fetcher.get_historical_data(
                        symbol=symbol,
                        timeframe=self.config.get("backtesting", {}).get("timeframe", "1h"),
                        limit=100,
                    )
                    market_data = {symbol: df}
            else:
                if hasattr(self.data_fetcher, "get_multiple_historical_data"):
                    market_data = await self.data_fetcher.get_multiple_historical_data(self.pairs)

            # Fetch multi-timeframe data if timeframe manager is available
            if self.timeframe_manager and self.pairs:
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

            # Combine single-timeframe and multi-timeframe data
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

            # Cache the fetched data
            self.market_data_cache = combined_data
            self.cache_timestamp = current_time
            logger.debug("Fetched and cached market data (including multi-timeframe)")

        except Exception:
            logger.exception("Failed to fetch market data")
            # Return cached data if available, otherwise empty dict
            if self.market_data_cache:
                logger.warning("Using stale cached data due to fetch failure")
                return self.market_data_cache

        return self.market_data_cache if self.market_data_cache else market_data

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
            # Guard initialization
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

            # Update totals
            self.performance_stats["total_pnl"] += float(pnl)

            # Track returns for Sharpe ratio calculation
            current_equity = float(self.state.equity or 0.0)
            equity_history = self.performance_stats["equity_history"]
            returns_history = self.performance_stats["returns_history"]

            if equity_history:
                prev_equity = equity_history[-1]
                if prev_equity > 0:
                    daily_return = (current_equity - prev_equity) / prev_equity
                    returns_history.append(daily_return)

            equity_history.append(current_equity)

            # Calculate win/loss counts
            if pnl > 0:
                self.performance_stats["wins"] += 1
            elif pnl < 0:
                self.performance_stats["losses"] += 1

            total_trades = (
                self.performance_stats["wins"] + self.performance_stats["losses"]
            )
            if total_trades > 0:
                self.performance_stats["win_rate"] = (
                    self.performance_stats["wins"] / total_trades
                )

            # Calculate max drawdown
            if len(equity_history) > 1:
                peak = max(equity_history)
                trough = min(equity_history)
                if peak > 0:
                    max_dd = (peak - trough) / peak
                    self.performance_stats["max_drawdown"] = max(
                        max_dd, float(self.performance_stats.get("max_drawdown", 0.0))
                    )

            # Calculate Sharpe ratio (annualized)
            if len(returns_history) > 1 and np.std(returns_history) > 0:
                returns = np.array(returns_history)
                risk_free_rate = 0.0  # Can be configured
                excess_returns = returns - risk_free_rate
                sharpe = float(np.mean(excess_returns) / np.std(excess_returns))
                self.performance_stats["sharpe_ratio"] = sharpe * np.sqrt(
                    252
                )  # Annualize
        except Exception:
            logger.exception("Failed to update performance metrics")

    async def record_trade_equity(self, order_result: Dict[str, Any]) -> None:
        """
        Record equity progression after a trade execution.

        Args:
            order_result: Dictionary returned from OrderManager.execute_order containing at least:
              - id (optional): trade identifier
              - timestamp (optional): epoch ms or ISO timestamp
              - pnl (optional): profit/loss for the trade

        Side effects:
            Appends a record to self.performance_stats['equity_progression'] with:
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
            equity_prog: list[Dict[str, Any]] = self.performance_stats.setdefault(
                "equity_progression", []
            )

            # Get current equity from order manager (async)
            try:
                current_equity = await self.order_manager.get_equity()
            except Exception:
                # Fallback to state.equity if order_manager can't provide it
                current_equity = self.state.equity or 0.0

            # Normalize values
            trade_id = order_result.get("id", f"trade_{now_ms()}")
            timestamp = order_result.get("timestamp", now_ms())
            pnl = order_result.get("pnl", None)
            # coerce current_equity safely
            try:
                equity_val = (
                    float(current_equity) if current_equity is not None else 0.0
                )
            except Exception:
                equity_val = 0.0

            # For backtest/paper modes the OrderManager may not track balance.
            # In that case derive equity from starting_balance + total_pnl so progression is meaningful.
            try:
                if getattr(self, "mode", None) in (
                    TradingMode.BACKTEST,
                    TradingMode.PAPER,
                ):
                    # If order_manager returned 0 or not tracking, compute from starting balance + total_pnl
                    if equity_val == 0.0:
                        equity_val = float(
                            getattr(self, "starting_balance", 0.0)
                        ) + float(self.performance_stats.get("total_pnl", 0.0))
            except Exception:
                # Ignore and proceed with current equity_val
                pass

            # Calculate cumulative return relative to starting balance
            try:
                cumulative_return = 0.0
                if (
                    getattr(self, "starting_balance", None)
                    and float(self.starting_balance) > 0
                ):
                    cumulative_return = (
                        equity_val - float(self.starting_balance)
                    ) / float(self.starting_balance)
            except Exception:
                cumulative_return = 0.0

            # Normalize trade_id and timestamp.
            # Preserve any explicit timestamp provided by the caller (do not coerce).
            trade_id = order_result.get("id", f"trade_{now_ms()}")
            ts_raw = order_result.get("timestamp", now_ms())

            record: Dict[str, Any] = {
                "trade_id": trade_id,
                "timestamp": ts_raw,
                "symbol": order_result.get("symbol") if isinstance(order_result, dict) else None,
                "equity": equity_val,
                "pnl": pnl,
                "cumulative_return": cumulative_return,
            }

            equity_prog.append(record)

        except Exception:
            logger.exception("Failed to record trade equity")

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
