"""
risk/risk_manager.py

Implements risk management controls including position sizing, stop-loss/take-profit,
and portfolio-level risk constraints. Validates all trading signals against risk rules.
"""

import logging
import asyncio
import random
from typing import Dict, Optional, Tuple, List, Any, Callable
from decimal import Decimal, getcontext, ROUND_HALF_UP
import math
import numpy as np
import pandas as pd
import time
from utils.time import now_ms, to_ms
from datetime import datetime, timedelta
from enum import Enum

from core.contracts import TradingSignal
from utils.config_loader import ConfigLoader
from utils.logger import get_trade_logger
from utils.adapter import signal_to_dict
from strategies.regime.market_regime import get_market_regime_detector, MarketRegime
from risk.anomaly_detector import get_anomaly_detector, AnomalyResponse
from risk.adaptive_policy import get_adaptive_risk_policy, get_risk_multiplier

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()

# Set decimal precision
getcontext().prec = 28  # keep high precision to avoid quantize errors


def _safe_quantize(value: Decimal, exp: Decimal = Decimal(".000001")) -> Decimal:
    """
    Safely quantize a Decimal to the given exponent. Fall back to string formatting
    when Decimal.quantize raises InvalidOperation.
    """
    # Handle NaN values first
    if value.is_nan():
        return Decimal("0")

    try:
        return value.quantize(exp, rounding=ROUND_HALF_UP)
    except (InvalidOperation, Exception):
        try:
            # Fallback: convert to float and format to 6 decimals then to Decimal
            return Decimal("{0:.6f}".format(float(value)))
        except Exception:
            return Decimal("0")


class RiskManager:
    """
    Manages all risk-related decisions including position sizing, stop-loss,
    take-profit, and portfolio-level risk constraints.
    """

    # NOTE (documentation):
    # - `signal.amount` is treated as a notional/quote-currency amount for the purposes
    #   of risk calculations and validation within this RiskManager. Tests and internal
    #   helpers expect `amount` to be comparable directly against the account balance
    #   (e.g., 1000 means 1,000 units of quote currency such as USD/USDT).
    #
    # - evaluate_signal will call calculate_position_size(...) when `signal.amount` is
    #   missing or zero. That calculation may return a raw computed amount based on the
    #   configured sizing method. To ensure portfolio-level risk limits are respected,
    #   evaluate_signal enforces a cap on the computed amount equal to:
    #       max_allowed = max_position_size * account_balance
    #   This guarantees no signal can request a position larger than the configured
    #   fraction of the account (even if the sizing routine itself produces a larger value).
    #
    # Rationale: This keeps sizing calculations flexible (used by unit tests and different
    # sizing methods) while making sure final accepted signals cannot exceed the global
    # risk limit (`max_position_size`).

    def __init__(self, config: Dict):
        """
        Initialize the RiskManager with configuration.

        Args:
            config: Risk management configuration dictionary
        """
        self.config = config
        self.require_stop_loss = config.get("require_stop_loss", True)
        self.max_position_size = Decimal(
            str(config.get("max_position_size", 0.3))
        )  # 30%
        self.max_daily_loss = Decimal(str(config.get("max_daily_drawdown", 0.1)))  # 10%
        self.risk_reward_ratio = Decimal(str(config.get("risk_reward_ratio", 2.0)))
        self.today_pnl = Decimal(0)
        self.today_start_balance = None
        self.position_sizing_method = config.get("position_sizing_method", "fixed")
        # Fixed percent sizing (fraction of account balance)
        self.fixed_percent = Decimal(str(config.get("fixed_percent", 0.1)))
        # Kelly criterion fallback assumptions
        self.kelly_assumed_win_rate = float(config.get("kelly_assumed_win_rate", 0.55))

        # Adaptive position sizing parameters
        self.risk_per_trade = Decimal(str(config.get("risk_per_trade", 0.02)))
        self.atr_k_factor = Decimal(str(config.get("atr_k_factor", 2.0)))

        # Dynamic stop loss parameters
        self.stop_loss_method = config.get("stop_loss_method", "atr")
        self.atr_sl_multiplier = Decimal(str(config.get("atr_sl_multiplier", 2.0)))
        self.stop_loss_percentage = Decimal(str(config.get("stop_loss_percentage", 0.02)))

        # Adaptive take profit parameters
        self.tp_base_multiplier = Decimal(str(config.get("tp_base_multiplier", 2.0)))
        self.enable_adaptive_tp = config.get("enable_adaptive_tp", True)

        # Trailing stop parameters
        self.enable_trailing_stop = config.get("enable_trailing_stop", True)
        self.trailing_stop_method = config.get("trailing_stop_method", "percentage")
        self.trailing_distance = Decimal(str(config.get("trailing_distance", 0.02)))
        self.trailing_atr_multiplier = Decimal(str(config.get("trailing_atr_multiplier", 1.5)))
        self.trailing_step_size = Decimal(str(config.get("trailing_step_size", 0.005)))

        # Time-based exit parameters
        self.enable_time_based_exit = config.get("ENABLE_TIME_EXIT", True)
        self.max_holding_candles = config.get("MAX_BARS_IN_TRADE", 50)
        self.timeframe = config.get("timeframe", "1h")

        # Regime-based exit parameters
        self.enable_regime_based_exit = config.get("enable_regime_based_exit", True)
        self.exit_on_regime_change = config.get("exit_on_regime_change", True)

        # Enhanced logging parameters
        self.enhanced_trade_logging = config.get("enhanced_trade_logging", True)
        self.track_exit_reasons = config.get("track_exit_reasons", True)
        self.log_sl_tp_details = config.get("log_sl_tp_details", True)

        # Initialize volatility tracker
        self.symbol_volatility = {}
        # Track per-symbol loss streaks for adaptive sizing
        self.loss_streaks = {}

        # Position tracking for trailing stops
        self.position_tracking = {}  # symbol -> tracking data

        # Exit type statistics
        self.exit_type_stats = {
            "sl_hit": {"wins": 0, "losses": 0},
            "tp_hit": {"wins": 0, "losses": 0},
            "time_limit": {"wins": 0, "losses": 0},
            "regime_change": {"wins": 0, "losses": 0},
            "manual": {"wins": 0, "losses": 0}
        }

        # Reliability / retry defaults (can be overridden via config["reliability"])
        rel_cfg = config.get("reliability", {}) if isinstance(config, dict) else {}
        self._reliability = {
            "max_retries": int(rel_cfg.get("max_retries", 2)),
            "backoff_base": float(rel_cfg.get("backoff_base", 0.25)),
            "max_backoff": float(rel_cfg.get("max_backoff", 5.0)),
            "safe_mode_threshold": int(rel_cfg.get("safe_mode_threshold", 10)),
            "block_on_errors": bool(rel_cfg.get("block_on_errors", False)),
        }
        # Track critical errors and blocking state
        self.critical_error_count = 0
        self.block_signals = False

    async def evaluate_signal(
        self, signal: TradingSignal, market_data: Dict = None
    ) -> bool:
        """
        Evaluate a trading signal against all risk rules.

        Args:
            signal: TradingSignal to evaluate
            market_data: Current market data for the symbol

        Returns:
            True if signal passes all risk checks, False otherwise
        """
        try:
            # Check basic signal validity
            if not await self._validate_signal_basics(signal):
                return False

            # Check for market anomalies
            if not await self._check_anomalies(signal, market_data):
                return False

            # Check portfolio-level risk constraints
            if not await self._check_portfolio_risk(signal):
                return False

            # Calculate position size if not provided
            if not hasattr(signal, "amount") or signal.amount <= 0:
                signal.amount = await self.calculate_position_size(signal, market_data)
                if signal.amount <= 0:
                    trade_logger.log_rejected_signal(signal_to_dict(signal), "zero_position_size")
                    return False

                # Enforce a cap based on max_position_size (as a fraction of account balance).
                # If the calculated amount exceeds the maximum allowed notional, cap it.
                try:
                    account_balance = await self._get_current_balance()
                    max_allowed = _safe_quantize(
                        self.max_position_size * account_balance
                    )
                    amt_dec = Decimal(str(signal.amount))
                    if amt_dec > max_allowed:
                        signal.amount = max_allowed
                except Exception:
                    # If any error occurs while capping, continue and let validation handle it.
                    pass

            # Validate position size
            if not await self._validate_position_size(signal):
                return False

            # Calculate stop loss if not provided
            if not signal.stop_loss:
                signal.stop_loss = await self.calculate_dynamic_stop_loss(signal, market_data)

            # Validate stop loss if required
            if self.require_stop_loss and not signal.stop_loss:
                trade_logger.log_rejected_signal(signal_to_dict(signal), "missing_stop_loss")
                return False

            # Calculate take profit if not provided
            if not signal.take_profit:
                signal.take_profit = await self.calculate_take_profit(signal)

            # Update volatility tracking
            if market_data and "close" in market_data:
                await self._update_volatility(signal.symbol, market_data["close"])

            return True

        except Exception as e:
            logger.error(f"Error evaluating signal: {str(e)}", exc_info=True)
            trade_logger.log_rejected_signal(signal_to_dict(signal), f"error: {str(e)}")
            return False

    async def calculate_position_size(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """
        Calculate appropriate position size based on configured sizing method and adaptive risk policy.

        Supported methods (configurable via risk_management.position_sizing_method):
          - adaptive_atr: ATR-based volatility scaling (NEW)
          - fixed_percent: allocate a fixed fraction of account balance (fixed_percent)
          - volatility: size based on ATR / volatility (uses market_data)
          - martingale: experimental doubling scheme (testing only)
          - kelly: position using simplified Kelly criterion (requires assumptions)
          - fixed (or any other): fixed fractional sizing (position_size in config)

        The calculated position size is then adjusted by the adaptive risk multiplier
        based on current market conditions and performance metrics.

        Args:
            signal: TradingSignal to calculate for
            market_data: Optional market data used by volatility sizing

        Returns:
            Position size in base currency (Decimal)
        """
        # Calculate base position size using selected method
        method = str(self.position_sizing_method).lower()
        if method == "adaptive_atr":
            base_position = await self.calculate_adaptive_position_size(signal, market_data)
        elif method == "volatility" or method == "volatility_based":
            base_position = await self._volatility_based_position_size(signal, market_data)
        elif method == "martingale":
            base_position = await self._martingale_position_size(signal)
        elif method == "kelly" or method == "kelly_criterion":
            base_position = await self._kelly_position_size(signal, market_data)
        elif method == "fixed_percent":
            # Use configured fixed_percent fraction of account balance
            account_balance = await self._get_current_balance()
            base_position = _safe_quantize(self.fixed_percent * account_balance)
        else:
            # Default: fixed fractional position sizing (legacy)
            base_position = await self._fixed_fractional_position_size(signal)

        # Apply adaptive risk multiplier
        risk_multiplier = await self._get_adaptive_risk_multiplier(signal.symbol, market_data)

        # Calculate final position size
        adjusted_position = base_position * Decimal(str(risk_multiplier))

        # Log the adjustment
        logger.info(
            f"Position size for {signal.symbol}: base={base_position:.2f}, "
            f"multiplier={risk_multiplier:.2f}, adjusted={adjusted_position:.2f}"
        )

        return _safe_quantize(adjusted_position)

    async def _fixed_fractional_position_size(self, signal: TradingSignal) -> Decimal:
        """
        Calculate position size using fixed fractional method.

        Args:
            signal: TradingSignal to calculate for

        Returns:
            Position size in base currency
        """
        account_balance = await self._get_current_balance()
        risk_percent = Decimal(
            str(self.config.get("position_size", 0.1))
        )  # Default 10%

        if signal.stop_loss and signal.current_price:
            stop_loss_pct = abs(
                (Decimal(str(signal.current_price)) - Decimal(str(signal.stop_loss)))
                / Decimal(str(signal.current_price))
            )
            risk_amount = account_balance * risk_percent
            position_size = risk_amount / stop_loss_pct
        else:
            position_size = account_balance * risk_percent

        return _safe_quantize(Decimal(position_size))

    async def _volatility_based_position_size(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """
        Calculate position size based on market volatility.

        Args:
            signal: TradingSignal to calculate for
            market_data: Optional current market data for the symbol

        Returns:
            Position size in base currency (Decimal)
        """
        if not market_data or "close" not in market_data:
            return await self._fixed_fractional_position_size(signal)

        closes = market_data["close"]
        if len(closes) < 20:  # Need enough data for volatility calculation
            return await self._fixed_fractional_position_size(signal)

        # Calculate ATR
        high_low = closes["high"] - closes["low"]
        high_close = np.abs(closes["high"] - closes["close"].shift())
        low_close = np.abs(closes["low"] - closes["close"].shift())
        true_range = np.maximum.reduce([high_low, high_close, low_close])
        atr = true_range.mean()

        # Handle NaN or zero ATR
        if atr <= 0 or np.isnan(atr):
            return await self._fixed_fractional_position_size(signal)

        account_balance = await self._get_current_balance()
        risk_amount = account_balance * Decimal(
            str(self.config.get("position_size", 0.1))
        )
        position_size = risk_amount / Decimal(str(atr))

        return _safe_quantize(Decimal(position_size))

    async def _martingale_position_size(self, signal: TradingSignal) -> Decimal:
        """
        Calculate position size using martingale method (for testing only).

        Args:
            signal: TradingSignal to calculate for

        Returns:
            Position size in base currency
        """
        # Note: This is for demonstration only - martingale is generally a bad strategy
        account_balance = await self._get_current_balance()
        loss_streak = await self._get_current_loss_streak(signal.symbol)

        if loss_streak == 0:
            return account_balance * Decimal("0.1")

        return account_balance * Decimal("0.1") * (Decimal("2") ** loss_streak)

    async def _kelly_position_size(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """
        Calculate position size using a simplified Kelly Criterion.

        This implementation uses an assumed win rate (config.kelly_assumed_win_rate)
        and the configured risk_reward_ratio to compute the Kelly fraction:

            k = w - (1 - w) / R

        where:
            w = assumed win rate
            R = average win / average loss approximated by risk_reward_ratio

        Returns:
            Position size in base currency (Decimal). Falls back to fixed fractional sizing
            if calculation is not feasible.
        """
        try:
            account_balance = await self._get_current_balance()
            w = float(self.kelly_assumed_win_rate)
            R = float(self.risk_reward_ratio) if self.risk_reward_ratio else 1.0
            k_fraction = max(0.0, (w - (1.0 - w) / R))
            # Cap Kelly to reasonable bounds and apply max_position_size
            k_fraction = min(k_fraction, float(self.max_position_size))
            position = _safe_quantize(Decimal(str(k_fraction)) * account_balance)
            if position <= 0:
                return await self._fixed_fractional_position_size(signal)
            return position
        except Exception:
            # Fallback to fixed fractional
            return await self._fixed_fractional_position_size(signal)

    async def calculate_take_profit(self, signal: TradingSignal) -> Optional[Decimal]:
        """
        Calculate take profit price based on risk/reward ratio.

        Args:
            signal: TradingSignal to calculate for

        Returns:
            Take profit price as Decimal, or None if it cannot be computed.
        """
        if not signal.stop_loss or not signal.current_price:
            return None

        entry = Decimal(str(signal.current_price))
        stop_loss = Decimal(str(signal.stop_loss))

        if signal.signal_type.name.endswith("LONG"):
            risk = entry - stop_loss
            take_profit = entry + risk * self.risk_reward_ratio
        else:  # SHORT
            risk = stop_loss - entry
            take_profit = entry - risk * self.risk_reward_ratio

        tp_dec = _safe_quantize(Decimal(take_profit))
        return tp_dec if tp_dec is not None else None

    async def _validate_signal_basics(self, signal: TradingSignal) -> bool:
        """Validate basic signal properties."""
        if not signal or not signal.symbol:
            trade_logger.log_rejected_signal(signal_to_dict(signal), "invalid_signal")
            return False

        if not signal.signal_type or not signal.order_type:
            trade_logger.log_rejected_signal(signal_to_dict(signal), "missing_type")
            return False

        return True

    async def _check_anomalies(self, signal: TradingSignal, market_data: Optional[Dict] = None) -> bool:
        """
        Check for market anomalies that should prevent trade execution.

        Args:
            signal: TradingSignal to check
            market_data: Current market data

        Returns:
            True if no anomalies detected, False if trade should be blocked
        """
        try:
            if not market_data:
                return True  # No market data available, allow trade

            # Get anomaly detector
            anomaly_detector = get_anomaly_detector()

            # Convert market data to DataFrame for anomaly detection
            if isinstance(market_data, dict):
                # Convert dict to DataFrame if needed
                data_df = pd.DataFrame(market_data)
            else:
                data_df = market_data

            # Check for anomalies
            should_proceed, response, anomaly = anomaly_detector.check_signal_anomaly(
                signal_to_dict(signal), data_df, signal.symbol
            )

            if not should_proceed:
                # Log the rejection reason
                reason = f"anomaly_{anomaly.anomaly_type.value}_{anomaly.severity.value}" if anomaly else "anomaly_detected"
                trade_logger.log_rejected_signal(signal_to_dict(signal), reason)

                # Apply response mechanism
                if response == AnomalyResponse.SCALE_DOWN:
                    # Scale down position size
                    if hasattr(signal, 'amount') and signal.amount:
                        original_amount = signal.amount
                        scale_factor = anomaly_detector.scale_down_factor
                        signal.amount = Decimal(str(signal.amount)) * Decimal(str(scale_factor))
                        logger.info(f"Scaled down position size from {original_amount} to {signal.amount} due to anomaly")

                return should_proceed

            return True

        except Exception as e:
            logger.warning(f"Error checking for anomalies: {e}")
            # On error, allow trade to proceed (fail-safe)
            return True

    async def _check_portfolio_risk(self, signal: TradingSignal) -> bool:
        """Check portfolio-level risk constraints."""
        # Check daily loss limit
        if self.today_start_balance and self.today_pnl < 0:
            drawdown_pct = abs(self.today_pnl) / self.today_start_balance
            if drawdown_pct >= self.max_daily_loss:
                trade_logger.log_rejected_signal(signal_to_dict(signal), "daily_loss_limit")
                return False

        # Check maximum concurrent positions
        current_positions = await self._get_current_positions()
        max_positions = self.config.get("max_concurrent_trades", 5)
        if len(current_positions) >= max_positions:
            trade_logger.log_rejected_signal(signal_to_dict(signal), "max_positions")
            return False

        return True

    async def _validate_position_size(self, signal: TradingSignal) -> bool:
        """Validate position size against risk rules."""
        account_balance = await self._get_current_balance()
        position_size_pct = Decimal(str(signal.amount)) / account_balance

        if position_size_pct > self.max_position_size:
            trade_logger.log_rejected_signal(signal_to_dict(signal), "position_too_large")
            return False

        return True

    async def _update_volatility(self, symbol: str, price_data: pd.Series) -> None:
        """Update volatility tracking for a symbol."""
        # Allow short series as tests use small series; require at least 2 points
        if len(price_data) < 2:
            return

        returns = np.log(price_data / price_data.shift(1)).dropna()
        if returns.empty:
            return

        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Don't store NaN values
        if np.isnan(volatility):
            return

        self.symbol_volatility[symbol] = {
            "volatility": float(volatility),
            "last_updated": now_ms(),
        }

    async def _get_current_balance(self) -> Decimal:
        """Get current account balance (simulated for backtesting)."""
        # In a real implementation, this would fetch from the exchange or portfolio manager
        return Decimal("10000")  # Mock balance for demonstration

    async def _get_current_positions(self) -> List[Any]:
        """Get current open positions (simulated for backtesting).

        Returns:
            List of current positions (mocked for tests).
        """
        # In a real implementation, this would fetch from the order manager
        return []  # Mock positions for demonstration

    async def _get_current_loss_streak(self, symbol: str) -> int:
        """Get current losing streak for a symbol (simulated for backtesting)."""
        # In a real implementation, this would analyze trade history
        return 0  # Mock streak for demonstration

    async def update_trade_outcome(
        self, symbol: str, pnl: Decimal, is_win: bool, timestamp: int = None
    ) -> None:
        """
        Update risk manager with trade outcome for adaptive calculations.

        This updates today's PnL, maintains a simple loss-streak counter per
        symbol and emits a small performance update to the trade logger.

        Args:
            symbol: Trading symbol
            pnl: Profit/loss from the trade
            is_win: Whether the trade was profitable
            timestamp: Time of the trade (optional)
        """
        # Update aggregate PnL
        try:
            self.today_pnl += Decimal(str(pnl))
        except Exception:
            # Ensure we always handle numeric-like inputs
            try:
                self.today_pnl += Decimal(pnl)
            except Exception:
                logger.exception("Invalid pnl value provided to update_trade_outcome")
                return

        # Normalize timestamp (store as epoch milliseconds)
        if timestamp is None:
            timestamp = now_ms()
        else:
            # Normalize any provided timestamp to ms if possible
            normalized = to_ms(timestamp)
            timestamp = normalized if normalized is not None else now_ms()

        # Update loss/win streak per symbol
        prev_streak = int(self.loss_streaks.get(symbol, 0))
        if is_win:
            # Reset the loss streak on a win
            self.loss_streaks[symbol] = 0
        else:
            # Increment loss streak on losing trade
            self.loss_streaks[symbol] = prev_streak + 1

        # Emit lightweight performance record to trade logger
        try:
            trade_logger.performance(
                "Trade outcome",
                {
                    "symbol": symbol,
                    "pnl": float(pnl),
                    "is_win": bool(is_win),
                    "loss_streak": int(self.loss_streaks.get(symbol, 0)),
                    "timestamp": int(timestamp),
                },
            )
        except Exception:
            logger.exception("Failed to emit trade outcome to trade logger")

    async def reset_daily_stats(self, current_balance: Decimal = None) -> None:
        """
        Reset daily tracking statistics.

        Args:
            current_balance: Current account balance to use as reference
        """
        self.today_pnl = Decimal(0)
        self.today_start_balance = current_balance or await self._get_current_balance()

    async def emergency_check(self) -> bool:
        """
        Perform emergency risk checks (e.g., market crashes).

        Returns:
            True if emergency measures should be triggered
        """
        # Check for excessive daily loss
        if self.today_pnl < 0:
            loss_pct = abs(self.today_pnl) / self.today_start_balance
            if loss_pct >= self.max_daily_loss * Decimal("1.5"):  # 1.5x daily limit
                return True

        return False

    async def get_risk_parameters(self, symbol: str = None) -> Dict:
        """
        Get current risk parameters for a symbol or globally.

        Args:
            symbol: Optional symbol to get parameters for

        Returns:
            Dictionary of risk parameters
        """
        params = {
            "max_position_size": float(self.max_position_size),
            "max_daily_loss": float(self.max_daily_loss),
            "risk_reward_ratio": float(self.risk_reward_ratio),
            "position_sizing_method": self.position_sizing_method,
            "today_pnl": float(self.today_pnl),
            "today_drawdown": float(
                abs(self.today_pnl) / self.today_start_balance
                if self.today_start_balance
                else 0.0
            ),
        }

        if symbol and symbol in self.symbol_volatility:
            params["volatility"] = self.symbol_volatility[symbol]["volatility"]

        return params

    # ===== NEW ADAPTIVE RISK MANAGEMENT METHODS =====

    async def calculate_adaptive_position_size(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """
        Calculate adaptive position size based on ATR volatility scaling.

        Formula: position_size = (account_equity * risk_per_trade) / (ATR * k_factor)

        Args:
            signal: TradingSignal to calculate for
            market_data: Market data containing OHLCV data

        Returns:
            Position size in base currency (Decimal)
        """
        try:
            account_balance = await self._get_current_balance()
            risk_per_trade = Decimal(str(self.config.get("risk_per_trade", 0.02)))  # 2% default

            if not market_data or "close" not in market_data:
                # Fallback to fixed percentage
                return _safe_quantize(account_balance * risk_per_trade)

            # Calculate ATR
            atr = await self._calculate_atr(market_data)
            if atr <= 0:
                return _safe_quantize(account_balance * risk_per_trade)

            # Get k-factor (volatility multiplier)
            k_factor = Decimal(str(self.config.get("atr_k_factor", 2.0)))

            # Calculate position size
            position_size = (account_balance * risk_per_trade) / (atr * k_factor)

            # Apply maximum position size constraint
            max_allowed = account_balance * self.max_position_size
            position_size = min(position_size, max_allowed)

            return _safe_quantize(position_size)

        except Exception as e:
            logger.warning(f"Error calculating adaptive position size: {e}")
            # Fallback to fixed percentage
            account_balance = await self._get_current_balance()
            risk_per_trade = Decimal(str(self.config.get("risk_per_trade", 0.02)))
            return _safe_quantize(account_balance * risk_per_trade)

    async def calculate_dynamic_stop_loss(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Decimal]:
        """
        Calculate dynamic stop loss based on ATR or percentage.

        Args:
            signal: TradingSignal to calculate for
            market_data: Market data for ATR calculation

        Returns:
            Stop loss price as Decimal, or None if cannot be computed
        """
        if not signal.current_price:
            return None

        entry_price = Decimal(str(signal.current_price))
        sl_method = self.stop_loss_method  # atr, percentage, fixed

        try:
            if sl_method == "atr":
                if not market_data:
                    return None
                atr = await self._calculate_atr(market_data)
                if atr <= 0:
                    return None

                atr_multiplier = Decimal(str(self.config.get("atr_sl_multiplier", 2.0)))
                atr_sl_distance = atr * atr_multiplier

                if signal.signal_type.name.endswith("LONG"):
                    return _safe_quantize(entry_price - atr_sl_distance)
                else:  # SHORT
                    return _safe_quantize(entry_price + atr_sl_distance)

            elif sl_method == "percentage":
                sl_percentage = self.stop_loss_percentage  # 2%
                sl_distance = entry_price * sl_percentage

                if signal.signal_type.name.endswith("LONG"):
                    return _safe_quantize(entry_price * (1 - sl_percentage))
                else:  # SHORT
                    return _safe_quantize(entry_price * (1 + sl_percentage))

            else:  # fixed
                # Use existing logic or fallback
                return await self._calculate_fixed_stop_loss(signal)

        except Exception as e:
            logger.warning(f"Error calculating dynamic stop loss: {e}")
            return None

    async def calculate_adaptive_take_profit(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Decimal]:
        """
        Calculate adaptive take profit based on risk multiples and trend strength.

        Args:
            signal: TradingSignal to calculate for
            market_data: Market data for ADX calculation

        Returns:
            Take profit price as Decimal, or None if cannot be computed
        """
        if not signal.stop_loss or not signal.current_price:
            return None

        entry_price = Decimal(str(signal.current_price))
        stop_loss = Decimal(str(signal.stop_loss))

        # Calculate risk amount
        if signal.signal_type.name.endswith("LONG"):
            risk_amount = entry_price - stop_loss
        else:  # SHORT
            risk_amount = stop_loss - entry_price

        # Get TP multiplier (default 2R, 3R based on trend strength)
        base_multiplier = Decimal(str(self.config.get("tp_base_multiplier", 2.0)))

        # Adjust multiplier based on trend strength (ADX)
        trend_multiplier = await self._calculate_trend_multiplier(market_data)
        final_multiplier = base_multiplier * trend_multiplier

        # Calculate take profit
        if signal.signal_type.name.endswith("LONG"):
            take_profit = entry_price + (risk_amount * final_multiplier)
        else:  # SHORT
            take_profit = entry_price - (risk_amount * final_multiplier)

        return _safe_quantize(take_profit)

    async def calculate_trailing_stop(
        self, signal: TradingSignal, current_price: Decimal,
        highest_price: Optional[Decimal] = None, lowest_price: Optional[Decimal] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Decimal]:
        """
        Calculate trailing stop loss based on ATR or percentage.

        Args:
            signal: TradingSignal for the position
            current_price: Current market price
            highest_price: Highest price since entry (for LONG positions)
            lowest_price: Lowest price since entry (for SHORT positions)
            market_data: Market data for ATR calculation

        Returns:
            Trailing stop price as Decimal, or None if cannot be computed
        """
        try:
            trail_method = self.trailing_stop_method  # atr, percentage
            trail_distance = Decimal(str(self.config.get("trailing_distance", 0.02)))  # 2% default

            if trail_method == "atr":
                if not market_data:
                    return None
                atr = await self._calculate_atr(market_data)
                if atr <= 0:
                    return None

                atr_multiplier = Decimal(str(self.config.get("TRAIL_ATR_MULTIPLIER", 2.0)))
                trail_distance = atr * atr_multiplier

            # Calculate trailing stop
            if signal.signal_type.name.endswith("LONG"):
                if highest_price:
                    # Trail below the highest price
                    trailing_stop = highest_price * (1 - trail_distance)
                    # Don't move stop loss up, only down
                    if signal.stop_loss:
                        trailing_stop = max(trailing_stop, Decimal(str(signal.stop_loss)))
                    return _safe_quantize(trailing_stop)
            else:  # SHORT
                if lowest_price:
                    # Trail above the lowest price
                    trailing_stop = lowest_price * (1 + trail_distance)
                    # Don't move stop loss down, only up
                    if signal.stop_loss:
                        trailing_stop = min(trailing_stop, Decimal(str(signal.stop_loss)))
                    return _safe_quantize(trailing_stop)

            return None

        except Exception as e:
            logger.warning(f"Error calculating trailing stop: {e}")
            return None

    async def should_exit_time_based(
        self, entry_timestamp: int, current_timestamp: int,
        timeframe: str = "1h", max_candles: int = 72
    ) -> Tuple[bool, str]:
        """
        Check if position should be closed based on time criteria.

        Args:
            entry_timestamp: Entry timestamp in milliseconds
            current_timestamp: Current timestamp in milliseconds
            timeframe: Chart timeframe (e.g., '1h', '4h', '1d')
            max_candles: Maximum number of candles to hold position

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        try:
            # Calculate time difference in milliseconds
            time_diff_ms = current_timestamp - entry_timestamp

            # Convert timeframe to milliseconds
            timeframe_ms = self._timeframe_to_ms(timeframe)

            # Calculate number of candles elapsed
            candles_elapsed = time_diff_ms / timeframe_ms

            if candles_elapsed >= max_candles:
                return True, f"time_limit_{max_candles}_candles"

            return False, ""

        except Exception as e:
            logger.warning(f"Error checking time-based exit: {e}")
            return False, ""

    async def should_exit_regime_change(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Check if position should be closed due to market regime change.

        Args:
            signal: TradingSignal for the position
            market_data: Current market data

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        try:
            if not market_data:
                return False, ""

            # Get market regime detector
            regime_detector = get_market_regime_detector()
            regime_result = regime_detector.detect_regime(pd.DataFrame(market_data))

            # Exit trend positions when regime switches to sideways
            if signal.signal_type.name.endswith("LONG") or signal.signal_type.name.endswith("SHORT"):
                if regime_result.regime == MarketRegime.SIDEWAYS:
                    # Check if we were previously in trending regime
                    previous_regime = regime_result.previous_regime
                    if previous_regime == MarketRegime.TRENDING:
                        return True, f"regime_change_{previous_regime.value}_to_{regime_result.regime.value}"

            return False, ""

        except Exception as e:
            logger.warning(f"Error checking regime-based exit: {e}")
            return False, ""

    async def update_position_tracking(
        self, signal: TradingSignal, current_price: Decimal,
        highest_price: Optional[Decimal] = None, lowest_price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Update position tracking for trailing stops and exit conditions.

        Args:
            signal: TradingSignal for the position
            current_price: Current market price
            highest_price: Highest price since entry
            lowest_price: Lowest price since entry

        Returns:
            Updated tracking information
        """
        try:
            # Update highest/lowest prices
            if signal.signal_type.name.endswith("LONG"):
                if highest_price is None:
                    highest_price = current_price
                else:
                    highest_price = max(highest_price, current_price)
            else:  # SHORT
                if lowest_price is None:
                    lowest_price = current_price
                else:
                    lowest_price = min(lowest_price, current_price)

            # Calculate trailing stop
            trailing_stop = await self.calculate_trailing_stop(
                signal, current_price, highest_price, lowest_price
            )

            return {
                "highest_price": highest_price,
                "lowest_price": lowest_price,
                "trailing_stop": trailing_stop,
                "current_price": current_price,
                "last_updated": now_ms()
            }

        except Exception as e:
            logger.warning(f"Error updating position tracking: {e}")
            return {}

    async def log_trade_with_exit_details(
        self, signal: TradingSignal, exit_price: Decimal, exit_reason: str,
        pnl: Decimal, entry_price: Decimal, stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ) -> None:
        """
        Enhanced trade logging with SL/TP details and exit reason tracking.

        Args:
            signal: TradingSignal for the trade
            exit_price: Price at which position was closed
            exit_reason: Reason for exit (sl_hit, tp_hit, time_limit, regime_change, etc.)
            pnl: Profit/loss from the trade
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        try:
            trade_details = {
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.name if signal.signal_type else "UNKNOWN",
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "pnl": float(pnl),
                "exit_reason": exit_reason,
                "stop_loss": float(stop_loss) if stop_loss else None,
                "take_profit": float(take_profit) if take_profit else None,
                "timestamp": now_ms(),
                "position_size": float(signal.amount) if hasattr(signal, 'amount') else None
            }

            trade_logger.performance("Trade closed", trade_details)

            # Update win rate by exit type
            await self._update_exit_type_stats(exit_reason, pnl > 0)

        except Exception as e:
            logger.exception("Failed to log trade with exit details")

    # ===== HELPER METHODS =====

    async def _calculate_atr(self, market_data: Dict[str, Any], period: int = 14) -> Decimal:
        """Calculate Average True Range from market data."""
        try:
            if "high" not in market_data or "low" not in market_data or "close" not in market_data:
                return Decimal("0")

            high = pd.Series(market_data["high"])
            low = pd.Series(market_data["low"])
            close = pd.Series(market_data["close"])

            if len(high) < period + 1:
                return Decimal("0")

            # Calculate True Range
            hl = high - low
            hc = (high - close.shift(1)).abs()
            lc = (low - close.shift(1)).abs()
            tr_values = np.maximum.reduce([hl.values, hc.values, lc.values])
            tr = pd.Series(tr_values, index=high.index)

            # Calculate ATR
            atr = tr.rolling(window=period).mean().iloc[-1]

            if np.isnan(atr) or atr <= 0:
                return Decimal("0")

            return Decimal(str(atr))

        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return Decimal("0")

    async def _calculate_trend_multiplier(self, market_data: Optional[Dict[str, Any]] = None) -> Decimal:
        """Calculate trend strength multiplier based on ADX."""
        try:
            if not market_data:
                return Decimal("1.0")

            # Get ADX value (would need to be calculated or provided in market_data)
            adx_value = market_data.get("adx", 25)  # Default to neutral

            # Higher ADX = stronger trend = higher TP multiplier
            if adx_value >= 40:  # Strong trend
                return Decimal("1.5")
            elif adx_value >= 25:  # Moderate trend
                return Decimal("1.2")
            else:  # Weak trend
                return Decimal("0.8")

        except Exception:
            return Decimal("1.0")

    async def _calculate_fixed_stop_loss(self, signal: TradingSignal) -> Optional[Decimal]:
        """Calculate fixed stop loss as fallback."""
        if not signal.current_price:
            return None

        entry_price = Decimal(str(signal.current_price))
        sl_percentage = Decimal(str(self.config.get("stop_loss_percentage", 0.02)))

        if signal.signal_type.name.endswith("LONG"):
            return _safe_quantize(entry_price * (1 - sl_percentage))
        else:  # SHORT
            return _safe_quantize(entry_price * (1 + sl_percentage))

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        timeframe = timeframe.lower()

        if timeframe == "1m":
            return 60 * 1000
        elif timeframe == "5m":
            return 5 * 60 * 1000
        elif timeframe == "15m":
            return 15 * 60 * 1000
        elif timeframe == "30m":
            return 30 * 60 * 1000
        elif timeframe == "1h":
            return 60 * 60 * 1000
        elif timeframe == "4h":
            return 4 * 60 * 60 * 1000
        elif timeframe == "1d":
            return 24 * 60 * 60 * 1000
        else:
            # Default to 1 hour
            return 60 * 60 * 1000

    async def _update_exit_type_stats(self, exit_reason: str, is_win: bool) -> None:
        """Update statistics for exit types."""
        try:
            if exit_reason in self.exit_type_stats:
                if is_win:
                    self.exit_type_stats[exit_reason]["wins"] += 1
                else:
                    self.exit_type_stats[exit_reason]["losses"] += 1
            else:
                # Initialize new exit reason
                self.exit_type_stats[exit_reason] = {"wins": 1 if is_win else 0, "losses": 0 if is_win else 1}

            logger.info(f"Exit type '{exit_reason}' resulted in {'win' if is_win else 'loss'}")
        except Exception:
            logger.exception("Failed to update exit type statistics")

    async def _get_adaptive_risk_multiplier(
        self, symbol: str, market_data: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Get adaptive risk multiplier from the adaptive risk policy.

        Args:
            symbol: Trading symbol
            market_data: Current market data

        Returns:
            Risk multiplier (0.1 to 1.0)
        """
        try:
            # Convert market data to DataFrame if needed
            if market_data and isinstance(market_data, dict):
                # Convert dict to DataFrame for the adaptive policy
                data_df = pd.DataFrame()
                for key, values in market_data.items():
                    if isinstance(values, (list, np.ndarray)):
                        data_df[key] = values
                    else:
                        # Single value, create series
                        data_df[key] = [values] * 20  # Pad with same value
            elif market_data is not None:
                data_df = market_data
            else:
                data_df = None

            # Get risk multiplier from adaptive policy
            multiplier, reasoning = get_risk_multiplier(symbol, data_df)

            # Log the multiplier decision
            logger.info(f"Adaptive risk multiplier for {symbol}: {multiplier:.2f} - {reasoning}")

            return multiplier

        except Exception as e:
            logger.warning(f"Error getting adaptive risk multiplier for {symbol}: {e}")
            # Return neutral multiplier on error
            return 1.0
