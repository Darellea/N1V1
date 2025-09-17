"""
Stochastic Oscillator Strategy

A range-based momentum strategy that uses the Stochastic oscillator to identify
overbought and oversold conditions, generating signals for mean reversion.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator trading strategy."""

    def __init__(self, config) -> None:
        """Initialize Stochastic strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "k_period": 14,  # %K period
            "d_period": 3,   # %D period (SMA of %K)
            "overbought": 80,  # Overbought threshold
            "oversold": 20,   # Oversold threshold
            "position_size": 0.08,  # 8% of portfolio
            "stop_loss_pct": 0.025,  # 2.5% stop loss
            "take_profit_pct": 0.05,  # 5% take profit
            "volume_filter": True,  # Use volume confirmation
            "volume_threshold": 1.2,  # Volume must be 1.2x average
            "divergence_filter": False,  # Use divergence confirmation
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic indicators for each symbol."""
        k_period = int(self.params["k_period"])
        d_period = int(self.params["d_period"])

        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_stochastic(group):
                # Calculate %K
                lowest_low = group["low"].rolling(window=k_period).min()
                highest_high = group["high"].rolling(window=k_period).max()
                group["stoch_k"] = 100 * (group["close"] - lowest_low) / (highest_high - lowest_low)

                # Calculate %D (SMA of %K)
                group["stoch_d"] = group["stoch_k"].rolling(window=d_period).mean()

                return group

            data = grouped.apply(calculate_stochastic).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            data = data.copy()
            lowest_low = data["low"].rolling(window=k_period).min()
            highest_high = data["high"].rolling(window=k_period).max()
            data["stoch_k"] = 100 * (data["close"] - lowest_low) / (highest_high - lowest_low)
            data["stoch_d"] = data["stoch_k"].rolling(window=d_period).mean()

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on Stochastic oscillator."""
        signals = []

        if isinstance(data, dict):
            for symbol, df in data.items():
                if df is not None and not df.empty:
                    signals.extend(await self._generate_signals_for_symbol(symbol, df))
        elif hasattr(data, 'groupby') and not data.empty and "symbol" in data.columns:
            grouped = data.groupby("symbol")
            for symbol, group in grouped:
                signals.extend(await self._generate_signals_for_symbol(symbol, group))
        else:
            try:
                df = pd.DataFrame(data)
                if not df.empty and "symbol" in df.columns:
                    grouped = df.groupby("symbol")
                    for symbol, group in grouped:
                        signals.extend(await self._generate_signals_for_symbol(symbol, group))
            except Exception:
                pass

        return signals

    async def _generate_signals_for_symbol(self, symbol: str, data) -> List[TradingSignal]:
        """Generate signals for a specific symbol's data."""
        signals = []

        try:
            # Calculate Stochastic for this symbol's data
            data_with_stoch = await self.calculate_indicators(data)

            if data_with_stoch.empty or len(data_with_stoch) < 2:
                return signals

            last_row = data_with_stoch.iloc[-1]
            prev_row = data_with_stoch.iloc[-2]
            current_price = last_row["close"]

            # Check for volume confirmation if enabled
            volume_confirmed = True
            if self.params["volume_filter"] and "volume" in last_row.index:
                try:
                    volume_period = int(self.params["k_period"])
                    if len(data_with_stoch) >= volume_period:
                        avg_volume = data_with_stoch["volume"].tail(volume_period).mean()
                        current_volume = last_row["volume"]
                        volume_confirmed = current_volume >= (avg_volume * self.params["volume_threshold"])
                except Exception:
                    volume_confirmed = True

            if not volume_confirmed:
                return signals

            # Check for oversold condition (%K and %D below oversold threshold)
            oversold_threshold = self.params["oversold"]
            oversold = (
                last_row["stoch_k"] < oversold_threshold and
                last_row["stoch_d"] < oversold_threshold and
                not pd.isna(last_row["stoch_k"]) and
                not pd.isna(last_row["stoch_d"])
            )

            # Check for overbought condition (%K and %D above overbought threshold)
            overbought_threshold = self.params["overbought"]
            overbought = (
                last_row["stoch_k"] > overbought_threshold and
                last_row["stoch_d"] > overbought_threshold and
                not pd.isna(last_row["stoch_k"]) and
                not pd.isna(last_row["stoch_d"])
            )

            # Additional filter: %K crossing %D for confirmation
            k_crossing_d_up = (
                prev_row["stoch_k"] <= prev_row["stoch_d"] and
                last_row["stoch_k"] > last_row["stoch_d"]
            )

            k_crossing_d_down = (
                prev_row["stoch_k"] >= prev_row["stoch_d"] and
                last_row["stoch_k"] < last_row["stoch_d"]
            )

            if oversold and k_crossing_d_up:
                # Stochastic is oversold and %K is crossing above %D
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_stoch.empty and isinstance(data_with_stoch.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_stoch.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,
                        strength=SignalStrength.MODERATE,
                        order_type="market",
                        amount=self.params["position_size"],
                        current_price=current_price,
                        stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                        take_profit=current_price * (1 + self.params["take_profit_pct"]),
                        metadata={
                            "oscillator_type": "stochastic",
                            "condition": "oversold",
                            "stoch_k": last_row["stoch_k"],
                            "stoch_d": last_row["stoch_d"],
                            "oversold_threshold": oversold_threshold,
                            "k_crossing_d": True
                        },
                        timestamp=signal_timestamp,
                    )
                )

            elif overbought and k_crossing_d_down:
                # Stochastic is overbought and %K is crossing below %D
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_stoch.empty and isinstance(data_with_stoch.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_stoch.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,
                        strength=SignalStrength.MODERATE,
                        order_type="market",
                        amount=self.params["position_size"],
                        current_price=current_price,
                        stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                        take_profit=current_price * (1 - self.params["take_profit_pct"]),
                        metadata={
                            "oscillator_type": "stochastic",
                            "condition": "overbought",
                            "stoch_k": last_row["stoch_k"],
                            "stoch_d": last_row["stoch_d"],
                            "overbought_threshold": overbought_threshold,
                            "k_crossing_d": True
                        },
                        timestamp=signal_timestamp,
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
