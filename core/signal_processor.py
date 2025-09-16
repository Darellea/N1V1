"""
SignalProcessor - Signal generation and risk evaluation component.

Handles strategy management, signal generation, risk assessment,
and signal routing through the risk management system.
"""

from typing import Dict, Any, Optional, List

from .logging_utils import get_structured_logger, LogSensitivity
from .utils.error_utils import ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory, CircuitBreaker

logger = get_structured_logger("core.signal_processor", LogSensitivity.SECURE)
error_handler = ErrorHandler("signal_processor")


class SignalProcessor:
    """
    Processes trading signals from strategies and evaluates them through risk management.

    Responsibilities:
    - Strategy initialization and management
    - Signal generation coordination
    - Risk evaluation and filtering
    - Strategy selector integration
    - Multi-timeframe signal processing
    """

    def __init__(self, config: Dict[str, Any], risk_manager=None, strategy_selector=None):
        """Initialize the SignalProcessor.

        Args:
            config: Configuration dictionary
            risk_manager: RiskManager instance for signal evaluation
            strategy_selector: Optional strategy selector for dynamic strategy switching
        """
        self.config = config
        self.risk_manager = risk_manager
        self.strategy_selector = strategy_selector

        # Active trading strategies
        self.strategies: List[Any] = []

        # Trading pairs
        self.pairs: List[str] = []

        # Safe mode flags
        self.block_signals: bool = False
        self.risk_block_signals: bool = False

        # Circuit breaker for external service calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("circuit_breaker", {}).get("failure_threshold", 5),
            recovery_timeout=config.get("circuit_breaker", {}).get("recovery_timeout", 60.0)
        )

        # Circuit breaker status
        self.circuit_breaker_status = "closed"

    def set_trading_pairs(self, pairs: List[str]):
        """Set the trading pairs for signal processing."""
        self.pairs = pairs
        logger.info(f"SignalProcessor configured for {len(pairs)} pairs")

    def add_strategy(self, strategy):
        """Add a trading strategy to the processor."""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.__class__.__name__}")

    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate trading signals from all active strategies."""
        # Input validation
        if not isinstance(market_data, dict):
            raise ValueError("market_data must be a dictionary")
        if not market_data:
            logger.warning("Empty market_data provided")
            return []

        signals = []

        if not self.strategies:
            logger.warning("No strategies available for signal generation")
            return signals

        # Use strategy selector if available and enabled
        if self.strategy_selector and self.strategy_selector.enabled and market_data:
            selected_signals = await self._generate_signals_with_selector(market_data)
            signals.extend(selected_signals)
        else:
            # Generate signals from all strategies
            all_signals = await self._generate_signals_from_all_strategies(market_data)
            signals.extend(all_signals)

        logger.debug(f"Generated {len(signals)} signals from strategies")
        return signals

    async def _generate_signals_with_selector(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate signals using strategy selector for dynamic strategy switching."""
        signals = []
        primary_symbol = list(market_data.keys())[0] if market_data else None

        if not primary_symbol or primary_symbol not in market_data:
            logger.warning("No primary symbol found for strategy selection")
            return await self._generate_signals_from_all_strategies(market_data)

        try:
            selected_strategy_class = self.strategy_selector.select_strategy(market_data[primary_symbol])

            if selected_strategy_class:
                selected_strategy = None
                for strategy in self.strategies:
                    if type(strategy) == selected_strategy_class:
                        selected_strategy = strategy
                        break

                if selected_strategy:
                    logger.info(f"Strategy selector chose: {selected_strategy_class.__name__}")
                    multi_tf_data = self._extract_multi_tf_data(market_data, primary_symbol)
                    strategy_signals = await selected_strategy.generate_signals(market_data, multi_tf_data)
                    signals.extend(strategy_signals)
                else:
                    logger.warning(f"Selected strategy {selected_strategy_class.__name__} not found in active strategies")
                    signals = await self._generate_signals_from_all_strategies(market_data)
            else:
                logger.warning("Strategy selector returned no strategy, using all available strategies")
                signals = await self._generate_signals_from_all_strategies(market_data)

        except (AttributeError, TypeError, KeyError) as e:
            logger.error(f"Strategy selection error - invalid data structure: {e}")
            signals = await self._generate_signals_from_all_strategies(market_data)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Strategy selection error - configuration or runtime issue: {e}")
            signals = await self._generate_signals_from_all_strategies(market_data)
        except Exception as e:
            logger.exception(f"Unexpected error in strategy selection: {e}")
            signals = await self._generate_signals_from_all_strategies(market_data)

        return signals

    async def _generate_signals_from_all_strategies(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate signals from all strategies when selector is disabled or fails."""
        signals = []

        for strategy in self.strategies:
            try:
                # Extract multi-timeframe data for the strategy's primary symbol
                primary_symbol = list(market_data.keys())[0] if market_data else None
                multi_tf_data = self._extract_multi_tf_data(market_data, primary_symbol) if primary_symbol else None
                strategy_signals = await strategy.generate_signals(market_data, multi_tf_data)
                signals.extend(strategy_signals)
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"Signal generation error - invalid data or configuration in {strategy.__class__.__name__}: {e}")
                # Continue with other strategies
            except asyncio.TimeoutError as e:
                logger.error(f"Signal generation timeout in {strategy.__class__.__name__}: {e}")
                # Continue with other strategies
            except Exception as e:
                logger.exception(f"Unexpected error generating signals from {strategy.__class__.__name__}: {e}")
                # Continue with other strategies

        return signals

    async def evaluate_risk(self, signals: List[Any], market_data: Dict[str, Any]) -> List[Any]:
        """Evaluate signals through risk management and return approved signals."""
        if not self.risk_manager:
            logger.warning("No risk manager available, approving all signals")
            return signals

        if self.risk_block_signals:
            logger.warning("Risk block active, rejecting all signals")
            return []

        approved_signals = []

        for signal in signals:
            try:
                if await self.risk_manager.evaluate_signal(signal, market_data):
                    approved_signals.append(signal)
                else:
                    logger.debug(f"Signal rejected by risk manager: {signal}")
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"Risk evaluation error - invalid signal data or configuration: {e}")
                # Conservatively reject signal on error
            except asyncio.TimeoutError as e:
                logger.error(f"Risk evaluation timeout: {e}")
                # Conservatively reject signal on error
            except Exception as e:
                logger.exception(f"Unexpected error evaluating signal through risk manager: {e}")
                # Conservatively reject signal on error

        logger.debug(f"Risk evaluation: {len(approved_signals)}/{len(signals)} signals approved")
        return approved_signals

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

    def enable_safe_mode(self):
        """Enable safe mode - block all signals."""
        self.block_signals = True
        logger.warning("SignalProcessor safe mode enabled - blocking all signals")

    def disable_safe_mode(self):
        """Disable safe mode - allow signals through."""
        self.block_signals = False
        logger.info("SignalProcessor safe mode disabled - signals allowed")

    def enable_risk_block(self):
        """Enable risk-based signal blocking."""
        self.risk_block_signals = True
        logger.warning("Risk-based signal blocking enabled")

    def disable_risk_block(self):
        """Disable risk-based signal blocking."""
        self.risk_block_signals = False
        logger.info("Risk-based signal blocking disabled")

    def get_strategy_info(self) -> List[Dict[str, Any]]:
        """Get information about active strategies."""
        return [
            {
                "name": strategy.__class__.__name__,
                "type": type(strategy).__name__,
                "active": True
            }
            for strategy in self.strategies
        ]

    def get_signal_stats(self) -> Dict[str, Any]:
        """Get signal processing statistics."""
        return {
            "active_strategies": len(self.strategies),
            "safe_mode_active": self.block_signals,
            "risk_block_active": self.risk_block_signals,
            "strategy_selector_enabled": self.strategy_selector.enabled if self.strategy_selector else False
        }

    async def initialize_strategies(self, data_fetcher):
        """Initialize all strategies with the data fetcher."""
        for strategy in self.strategies:
            try:
                if hasattr(strategy, 'initialize'):
                    # Use circuit breaker for external service calls
                    await self.circuit_breaker.call(strategy.initialize, data_fetcher)
                logger.info(f"Initialized strategy: {strategy.__class__.__name__}")
            except (AttributeError, TypeError, ValueError) as e:
                logger.error(f"Strategy initialization error - invalid data fetcher or configuration in {strategy.__class__.__name__}: {e}")
            except asyncio.TimeoutError as e:
                logger.error(f"Strategy initialization timeout in {strategy.__class__.__name__}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error initializing strategy {strategy.__class__.__name__}: {e}")

    async def shutdown_strategies(self):
        """Shutdown all strategies."""
        for strategy in self.strategies:
            try:
                if hasattr(strategy, 'shutdown'):
                    await strategy.shutdown()
                logger.debug(f"Shutdown strategy: {strategy.__class__.__name__}")
            except (AttributeError, TypeError) as e:
                logger.error(f"Strategy shutdown error - invalid strategy state in {strategy.__class__.__name__}: {e}")
            except asyncio.TimeoutError as e:
                logger.error(f"Strategy shutdown timeout in {strategy.__class__.__name__}: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error shutting down strategy {strategy.__class__.__name__}: {e}")
