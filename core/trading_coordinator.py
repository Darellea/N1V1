"""
TradingCoordinator - Coordinates trading operations and manages execution flow.

Handles the main trading cycle, orchestrates data fetching, signal processing,
risk evaluation, and order execution while maintaining clean separation of concerns.
"""

from typing import Dict, Any, Optional, List
import asyncio
import time

from .logging_utils import get_structured_logger, LogSensitivity
from .utils.error_utils import ErrorHandler, ErrorContext, ErrorSeverity, ErrorCategory
from .utils.config_utils import get_config

logger = get_structured_logger("core.trading_coordinator", LogSensitivity.SECURE)
error_handler = ErrorHandler("trading_coordinator")


class TradingCoordinator:
    """
    Coordinates trading operations and manages the main execution flow.

    Responsibilities:
    - Orchestrate the main trading cycle
    - Coordinate data fetching and processing
    - Manage signal generation and risk evaluation
    - Handle order execution and performance tracking
    - Maintain safe mode and emergency shutdown procedures
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the TradingCoordinator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = config.get("environment", {}).get("mode", "paper")
        self.portfolio_mode = bool(config.get("trading", {}).get("portfolio_mode", False))
        self.pairs = []

        # Component references (to be injected)
        self.data_manager = None
        self.signal_processor = None
        self.risk_manager = None
        self.order_executor = None
        self.performance_tracker = None
        self.state_manager = None

        # Safe mode and control flags
        self.global_safe_mode = False
        self._safe_mode_notified = False

        # Trading cycle configuration
        self.update_interval = config.get("monitoring", {}).get("update_interval", 60)

    def set_components(self,
                      data_manager,
                      signal_processor,
                      risk_manager,
                      order_executor,
                      performance_tracker,
                      state_manager):
        """Set component references for coordination."""
        self.data_manager = data_manager
        self.signal_processor = signal_processor
        self.risk_manager = risk_manager
        self.order_executor = order_executor
        self.performance_tracker = performance_tracker
        self.state_manager = state_manager

    def set_trading_pairs(self, pairs: List[str]):
        """Set the trading pairs for coordination."""
        self.pairs = pairs
        logger.info(f"TradingCoordinator configured for {len(pairs)} pairs: {pairs}")

    async def execute_trading_cycle(self) -> None:
        """Execute one complete trading cycle."""
        try:
            # 1. Fetch market data
            market_data = await self._fetch_market_data()
            if not market_data:
                logger.debug("No market data available, skipping cycle")
                return

            # 2. Check safe mode conditions
            if await self._check_safe_mode_conditions():
                return

            # 3. Process through binary model integration (if enabled)
            integrated_decisions = await self._process_binary_integration(market_data)

            # 4. Generate signals from strategies (legacy path or when binary integration fails)
            if not integrated_decisions:
                signals = await self._generate_signals(market_data)
                if signals:
                    # 5. Route signals through risk management
                    approved_signals = await self._evaluate_risk(signals, market_data)
                    if approved_signals:
                        # 6. Execute orders
                        await self._execute_orders(approved_signals)
            else:
                # Execute integrated decisions
                await self._execute_integrated_decisions(integrated_decisions)

            # 7. Update bot state
            await self._update_state()

        except Exception as e:
            context = ErrorContext(
                component="trading_coordinator",
                operation="execute_trading_cycle",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.LOGIC,
                metadata={"cycle_error": str(e)}
            )
            await error_handler.handle_error(e, context)

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data through the data manager."""
        if not self.data_manager:
            logger.error("DataManager not set, cannot fetch market data")
            return {}

        try:
            return await self.data_manager.fetch_market_data()
        except Exception as e:
            logger.exception("Failed to fetch market data")
            return {}

    async def _check_safe_mode_conditions(self) -> bool:
        """Check various safe mode conditions and return True if trading should be skipped."""
        try:
            # Check global safe mode
            if self.global_safe_mode:
                if not self._safe_mode_notified:
                    logger.warning("Global safe mode active: skipping trading cycle")
                    self._safe_mode_notified = True
                return True

            # Check component safe modes
            if hasattr(self.order_executor, 'safe_mode_active') and self.order_executor.safe_mode_active:
                if not self._safe_mode_notified:
                    logger.warning("Order executor safe mode active: skipping trading cycle")
                    self._safe_mode_notified = True
                return True

            if hasattr(self.signal_processor, 'block_signals') and self.signal_processor.block_signals:
                if not self._safe_mode_notified:
                    logger.warning("Signal processor safe mode active: skipping trading cycle")
                    self._safe_mode_notified = True
                return True

            return False

        except Exception as e:
            logger.exception("Error checking safe mode conditions")
            return True  # Conservative: skip trading on error

    async def _process_binary_integration(self, market_data: Dict[str, Any]) -> List[Any]:
        """Process market data through binary model integration."""
        # This would integrate with binary model processing
        # For now, return empty list to use legacy signal processing
        return []

    async def _generate_signals(self, market_data: Dict[str, Any]) -> List[Any]:
        """Generate trading signals through the signal processor."""
        if not self.signal_processor:
            logger.error("SignalProcessor not set, cannot generate signals")
            return []

        try:
            return await self.signal_processor.generate_signals(market_data)
        except Exception as e:
            logger.exception("Failed to generate signals")
            return []

    async def _evaluate_risk(self, signals: List[Any], market_data: Dict[str, Any]) -> List[Any]:
        """Evaluate signals through risk management."""
        if not self.signal_processor:
            logger.warning("SignalProcessor not set, approving all signals")
            return signals

        try:
            return await self.signal_processor.evaluate_risk(signals, market_data)
        except Exception as e:
            logger.exception("Failed to evaluate risk")
            return []  # Conservative: reject all on error

    async def _execute_orders(self, approved_signals: List[Any]) -> None:
        """Execute approved trading signals."""
        if not self.order_executor:
            logger.error("OrderExecutor not set, cannot execute orders")
            return

        try:
            for signal in approved_signals:
                order_result = await self.order_executor.execute_order(signal)

                # Update performance metrics
                if order_result and "pnl" in order_result:
                    if self.performance_tracker:
                        await self.performance_tracker.record_trade_equity(order_result)

                    # Send notifications if available
                    if hasattr(self.order_executor, 'send_notifications'):
                        await self.order_executor.send_notifications(order_result)

        except Exception as e:
            logger.exception("Failed to execute orders")

    async def _execute_integrated_decisions(self, integrated_decisions: List[Dict[str, Any]]) -> None:
        """Execute integrated trading decisions from binary model."""
        # Implementation for binary model decisions
        pass

    async def _update_state(self) -> None:
        """Update the bot's internal state."""
        if not self.state_manager:
            return

        try:
            # Update state through state manager
            await self.state_manager.update_state()
        except Exception as e:
            logger.exception("Failed to update state")

    def enable_safe_mode(self):
        """Enable global safe mode."""
        self.global_safe_mode = True
        self._safe_mode_notified = False
        logger.warning("Global safe mode enabled")

    def disable_safe_mode(self):
        """Disable global safe mode."""
        self.global_safe_mode = False
        self._safe_mode_notified = False
        logger.info("Global safe mode disabled")

    async def emergency_shutdown(self) -> None:
        """Execute emergency shutdown procedures."""
        logger.critical("Executing emergency shutdown!")

        try:
            # Cancel all open orders
            if self.order_executor:
                await self.order_executor.cancel_all_orders()

            # Shutdown components
            if self.data_manager and hasattr(self.data_manager, 'shutdown'):
                await self.data_manager.shutdown()

            if self.signal_processor and hasattr(self.signal_processor, 'shutdown_strategies'):
                await self.signal_processor.shutdown_strategies()

        except Exception as e:
            logger.exception("Error during emergency shutdown")

    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get coordinator status information."""
        return {
            "global_safe_mode": self.global_safe_mode,
            "portfolio_mode": self.portfolio_mode,
            "trading_pairs": self.pairs,
            "update_interval": self.update_interval,
            "components_initialized": {
                "data_manager": self.data_manager is not None,
                "signal_processor": self.signal_processor is not None,
                "risk_manager": self.risk_manager is not None,
                "order_executor": self.order_executor is not None,
                "performance_tracker": self.performance_tracker is not None,
                "state_manager": self.state_manager is not None
            }
        }
