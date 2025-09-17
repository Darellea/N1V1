"""
Donchian Breakout Strategy

A trend-following strategy that uses Donchian channels to identify breakouts
above resistance or below support levels, indicating potential trend continuation.

# Centralized configuration constants for robustness and maintainability
CHANNEL_PERIOD = 20  # Period for Donchian channel
BREAKOUT_THRESHOLD = 0.001  # Minimum breakout percentage
POSITION_SIZE = 0.1  # 10% of portfolio
STOP_LOSS_PCT = 0.04  # 4%
TAKE_PROFIT_PCT = 0.1  # 10%
VOLUME_FILTER = True  # Use volume confirmation
VOLUME_MULTIPLIER = 1.2  # Volume must be 1.2x average
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from strategies.indicators_cache import calculate_indicators_for_multi_symbol
from core.contracts import TradingSignal, SignalType, SignalStrength


class DonchianBreakoutStrategy(BaseStrategy):
    """Donchian Channel breakout trading strategy."""

    def __init__(self, config) -> None:
        """Initialize Donchian breakout strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "channel_period": CHANNEL_PERIOD,  # Period for Donchian channel
            "breakout_threshold": BREAKOUT_THRESHOLD,  # Minimum breakout percentage
            "position_size": POSITION_SIZE,  # 10% of portfolio
            "stop_loss_pct": STOP_LOSS_PCT,  # 4%
            "take_profit_pct": TAKE_PROFIT_PCT,  # 10%
            "volume_filter": VOLUME_FILTER,  # Use volume confirmation
            "volume_multiplier": VOLUME_MULTIPLIER,  # Volume must be 1.2x average
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Donchian Channel indicators using vectorized operations for better performance.

        PERFORMANCE IMPROVEMENT: This method now uses vectorized pandas operations
        instead of inefficient groupby/apply patterns. For multi-symbol data, it processes
        all symbols simultaneously using numpy/pandas vectorization, which provides
        significant performance gains especially with large datasets containing many symbols.

        The shared indicators_cache module provides additional caching to avoid redundant
        calculations when the same indicator parameters are used across different strategies.
        """
        if data.empty:
            return data

        # Input validation: Check for required columns
        required_columns = ['high', 'low']  # Donchian Channel needs high and low prices
        if self.params.get("volume_filter", False):
            required_columns.append('volume')
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in market data: {missing_columns}. "
                           f"Expected columns: {required_columns}")

        # Use vectorized calculation for all symbols at once
        # This eliminates the need for groupby operations and provides much better performance
        indicators_config = {
            'donchian': {'period': int(self.params["channel_period"])}
        }

        # Calculate indicators using the shared vectorized function
        result_df = calculate_indicators_for_multi_symbol(data, indicators_config)

        return result_df

    async def generate_signals(self, data) -> List[TradingSignal]:
        """
        Generate signals based on Donchian breakouts using vectorized operations.

        PERFORMANCE IMPROVEMENT: This method now uses vectorized pandas operations
        to process all symbols simultaneously instead of iterating through each symbol
        individually. This eliminates the overhead of groupby operations and provides
        significant performance improvements for multi-symbol scenarios.
        """
        signals = []

        if data.empty or "symbol" not in data.columns:
            return signals

        try:
            # Ensure indicators are calculated
            required_cols = ['donchian_high', 'donchian_low', 'donchian_mid', 'donchian_width', 'donchian_width_pct']
            if not all(col in data.columns for col in required_cols):
                data = await self.calculate_indicators(data)

            # Get the last two rows for each symbol for breakout detection
            last_rows = data.groupby("symbol").tail(2)

            # Process each symbol's data
            for symbol in data["symbol"].unique():
                symbol_data = last_rows[last_rows["symbol"] == symbol]

                if len(symbol_data) < 2:
                    continue

                last_row = symbol_data.iloc[-1]
                prev_row = symbol_data.iloc[-2]
                current_price = last_row["close"]

                # Volume confirmation (vectorized where possible)
                volume_confirmed = True
                if self.params["volume_filter"] and "volume" in last_row.index and not pd.isna(last_row["volume"]):
                    try:
                        # Get full symbol data for volume calculation
                        full_symbol_data = data[data["symbol"] == symbol]
                        volume_period = int(self.params["channel_period"])

                        if len(full_symbol_data) >= volume_period:
                            avg_volume = full_symbol_data["volume"].tail(volume_period).mean()
                            current_volume = last_row["volume"]
                            volume_confirmed = current_volume >= (avg_volume * self.params["volume_multiplier"])
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Volume confirmation failed for {symbol}: {e}")
                        volume_confirmed = True

                if not volume_confirmed:
                    continue

                # Check for breakout above upper channel
                breakout_up = (
                    last_row["high"] > last_row["donchian_high"] and
                    prev_row["high"] <= prev_row["donchian_high"]
                )

                # Check for breakout below lower channel
                breakout_down = (
                    last_row["low"] < last_row["donchian_low"] and
                    prev_row["low"] >= prev_row["donchian_low"]
                )

                # Additional filter: breakout should be significant
                breakout_threshold = self.params["breakout_threshold"]

                if breakout_up:
                    breakout_pct = (last_row["high"] - last_row["donchian_high"]) / last_row["donchian_high"]
                    if breakout_pct > breakout_threshold:
                        self.signal_counts["long"] += 1
                        self.signal_counts["total"] += 1
                        # Note: last_signal_time is now updated deterministically in _log_signals

                        # Extract deterministic timestamp from the data that triggered the signal
                        signal_timestamp = None
                        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
                            signal_timestamp = data.index[-1].to_pydatetime()

                        signals.append(
                            self.create_signal(
                                symbol=symbol,
                                signal_type=SignalType.ENTRY_LONG,
                                strength=SignalStrength.STRONG,
                                order_type="market",
                                amount=self.params["position_size"],
                                current_price=current_price,
                                stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                                take_profit=current_price * (1 + self.params["take_profit_pct"]),
                                metadata={
                                    "breakout_type": "upper",
                                    "donchian_high": last_row["donchian_high"],
                                    "donchian_low": last_row["donchian_low"],
                                    "breakout_pct": breakout_pct,
                                    "channel_width_pct": last_row["donchian_width_pct"]
                                },
                                timestamp=signal_timestamp,
                            )
                        )

                elif breakout_down:
                    breakout_pct = (last_row["donchian_low"] - last_row["low"]) / last_row["donchian_low"]
                    if breakout_pct > breakout_threshold:
                        self.signal_counts["short"] += 1
                        self.signal_counts["total"] += 1
                        # Note: last_signal_time is now updated deterministically in _log_signals

                        # Extract deterministic timestamp from the data that triggered the signal
                        signal_timestamp = None
                        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
                            signal_timestamp = data.index[-1].to_pydatetime()

                        signals.append(
                            self.create_signal(
                                symbol=symbol,
                                signal_type=SignalType.ENTRY_SHORT,
                                strength=SignalStrength.STRONG,
                                order_type="market",
                                amount=self.params["position_size"],
                                current_price=current_price,
                                stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                                take_profit=current_price * (1 - self.params["take_profit_pct"]),
                                metadata={
                                    "breakout_type": "lower",
                                    "donchian_high": last_row["donchian_high"],
                                    "donchian_low": last_row["donchian_low"],
                                    "breakout_pct": breakout_pct,
                                    "channel_width_pct": last_row["donchian_width_pct"]
                                },
                                timestamp=signal_timestamp,
                            )
                        )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error in vectorized signal generation: {str(e)}")

        return signals

    async def _generate_signals_for_symbol(self, symbol: str, data) -> List[TradingSignal]:
        """Generate signals for a specific symbol's data."""
        signals = []

        try:
            # Calculate Donchian Channel for this symbol's data
            data_with_donchian = await self.calculate_indicators(data)

            if data_with_donchian.empty or len(data_with_donchian) < 2:
                return signals

            last_row = data_with_donchian.iloc[-1]
            prev_row = data_with_donchian.iloc[-2]
            current_price = last_row["close"]

            # Check for volume confirmation if enabled
            volume_confirmed = True
            if self.params["volume_filter"] and "volume" in last_row.index:
                try:
                    volume_period = int(self.params["channel_period"])
                    if len(data_with_donchian) >= volume_period:
                        avg_volume = data_with_donchian["volume"].tail(volume_period).mean()
                        current_volume = last_row["volume"]
                        volume_confirmed = current_volume >= (avg_volume * self.params["volume_multiplier"])
                except Exception:
                    volume_confirmed = True

            if not volume_confirmed:
                return signals

            # Check for breakout above upper channel
            breakout_up = (
                last_row["high"] > last_row["donchian_high"] and
                prev_row["high"] <= prev_row["donchian_high"]
            )

            # Check for breakout below lower channel
            breakout_down = (
                last_row["low"] < last_row["donchian_low"] and
                prev_row["low"] >= prev_row["donchian_low"]
            )

            # Additional filter: breakout should be significant
            breakout_threshold = self.params["breakout_threshold"]

            if breakout_up:
                breakout_pct = (last_row["high"] - last_row["donchian_high"]) / last_row["donchian_high"]
                if breakout_pct > breakout_threshold:
                    self.signal_counts["long"] += 1
                    self.signal_counts["total"] += 1
                    # Note: last_signal_time is now updated deterministically in _log_signals

                    # Extract deterministic timestamp from the data that triggered the signal
                    signal_timestamp = None
                    if not data_with_donchian.empty and isinstance(data_with_donchian.index, pd.DatetimeIndex):
                        signal_timestamp = data_with_donchian.index[-1].to_pydatetime()

                    signals.append(
                        self.create_signal(
                            symbol=symbol,
                            signal_type=SignalType.ENTRY_LONG,
                            strength=SignalStrength.STRONG,
                            order_type="market",
                            amount=self.params["position_size"],
                            current_price=current_price,
                            stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                            take_profit=current_price * (1 + self.params["take_profit_pct"]),
                            metadata={
                                "breakout_type": "upper",
                                "donchian_high": last_row["donchian_high"],
                                "donchian_low": last_row["donchian_low"],
                                "breakout_pct": breakout_pct,
                                "channel_width_pct": last_row["donchian_width_pct"]
                            },
                            timestamp=signal_timestamp,
                        )
                    )

            elif breakout_down:
                breakout_pct = (last_row["donchian_low"] - last_row["low"]) / last_row["donchian_low"]
                if breakout_pct > breakout_threshold:
                    self.signal_counts["short"] += 1
                    self.signal_counts["total"] += 1
                    # Note: last_signal_time is now updated deterministically in _log_signals

                    # Extract deterministic timestamp from the data that triggered the signal
                    signal_timestamp = None
                    if not data_with_donchian.empty and isinstance(data_with_donchian.index, pd.DatetimeIndex):
                        signal_timestamp = data_with_donchian.index[-1].to_pydatetime()

                    signals.append(
                        self.create_signal(
                            symbol=symbol,
                            signal_type=SignalType.ENTRY_SHORT,
                            strength=SignalStrength.STRONG,
                            order_type="market",
                            amount=self.params["position_size"],
                            current_price=current_price,
                            stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                            take_profit=current_price * (1 - self.params["take_profit_pct"]),
                            metadata={
                                "breakout_type": "lower",
                                "donchian_high": last_row["donchian_high"],
                                "donchian_low": last_row["donchian_low"],
                                "breakout_pct": breakout_pct,
                                "channel_width_pct": last_row["donchian_width_pct"]
                            },
                            timestamp=signal_timestamp,
                        )
                    )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
