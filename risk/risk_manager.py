"""
risk/risk_manager.py

Implements risk management controls including position sizing, stop-loss/take-profit,
and portfolio-level risk constraints. Validates all trading signals against risk rules.
"""

import logging
import asyncio
import random
from typing import Dict, Optional, Tuple, List, Any, Callable
from decimal import Decimal, getcontext, InvalidOperation, ROUND_HALF_UP
import math
import numpy as np
import pandas as pd
import time

from core.signal_router import TradingSignal
from utils.config_loader import ConfigLoader
from utils.logger import TradeLogger

logger = logging.getLogger(__name__)
trade_logger = TradeLogger()

# Set decimal precision
getcontext().prec = 28  # keep high precision to avoid quantize errors


def _safe_quantize(value: Decimal, exp: Decimal = Decimal(".000001")) -> Decimal:
    """
    Safely quantize a Decimal to the given exponent. Fall back to string formatting
    when Decimal.quantize raises InvalidOperation.
    """
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
        # max_position_size kept as Decimal above

        # Initialize volatility tracker
        self.symbol_volatility = {}
        # Track per-symbol loss streaks for adaptive sizing
        self.loss_streaks = {}

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

            # Check portfolio-level risk constraints
            if not await self._check_portfolio_risk(signal):
                return False

            # Validate stop loss if required
            if self.require_stop_loss and not signal.stop_loss:
                trade_logger.log_rejected_signal(signal, "missing_stop_loss")
                return False

            # Calculate position size if not provided
            if not hasattr(signal, "amount") or signal.amount <= 0:
                signal.amount = await self.calculate_position_size(signal, market_data)
                if signal.amount <= 0:
                    trade_logger.log_rejected_signal(signal, "zero_position_size")
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

            # Calculate take profit if not provided
            if not signal.take_profit:
                signal.take_profit = await self.calculate_take_profit(signal)

            # Update volatility tracking
            if market_data and "close" in market_data:
                await self._update_volatility(signal.symbol, market_data["close"])

            return True

        except Exception as e:
            logger.error(f"Error evaluating signal: {str(e)}", exc_info=True)
            trade_logger.log_rejected_signal(signal, f"error: {str(e)}")
            return False

    async def calculate_position_size(
        self, signal: TradingSignal, market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """
        Calculate appropriate position size based on configured sizing method.

        Supported methods (configurable via risk_management.position_sizing_method):
          - fixed_percent: allocate a fixed fraction of account balance (fixed_percent)
          - volatility: size based on ATR / volatility (uses market_data)
          - martingale: experimental doubling scheme (testing only)
          - kelly: position using simplified Kelly criterion (requires assumptions)
          - fixed (or any other): fixed fractional sizing (position_size in config)

        Args:
            signal: TradingSignal to calculate for
            market_data: Optional market data used by volatility sizing

        Returns:
            Position size in base currency (Decimal)
        """
        method = str(self.position_sizing_method).lower()
        if method == "volatility" or method == "volatility_based":
            return await self._volatility_based_position_size(signal, market_data)
        if method == "martingale":
            return await self._martingale_position_size(signal)
        if method == "kelly" or method == "kelly_criterion":
            return await self._kelly_position_size(signal, market_data)
        if method == "fixed_percent":
            # Use configured fixed_percent fraction of account balance
            account_balance = await self._get_current_balance()
            position = _safe_quantize(self.fixed_percent * account_balance)
            return position
        # Default: fixed fractional position sizing (legacy)
        return await self._fixed_fractional_position_size(signal)

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

        if atr <= 0:
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
            trade_logger.log_rejected_signal(signal, "invalid_signal")
            return False

        if not signal.signal_type or not signal.order_type:
            trade_logger.log_rejected_signal(signal, "missing_type")
            return False

        return True

    async def _check_portfolio_risk(self, signal: TradingSignal) -> bool:
        """Check portfolio-level risk constraints."""
        # Check daily loss limit
        if self.today_start_balance and self.today_pnl < 0:
            drawdown_pct = abs(self.today_pnl) / self.today_start_balance
            if drawdown_pct >= self.max_daily_loss:
                trade_logger.log_rejected_signal(signal, "daily_loss_limit")
                return False

        # Check maximum concurrent positions
        current_positions = await self._get_current_positions()
        max_positions = self.config.get("max_concurrent_trades", 5)
        if len(current_positions) >= max_positions:
            trade_logger.log_rejected_signal(signal, "max_positions")
            return False

        return True

    async def _validate_position_size(self, signal: TradingSignal) -> bool:
        """Validate position size against risk rules."""
        account_balance = await self._get_current_balance()
        position_size_pct = Decimal(str(signal.amount)) / account_balance

        if position_size_pct > self.max_position_size:
            trade_logger.log_rejected_signal(signal, "position_too_large")
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

        self.symbol_volatility[symbol] = {
            "volatility": float(volatility),
            "last_updated": time.time(),
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

        # Normalize timestamp
        if timestamp is None:
            timestamp = int(time.time())

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
