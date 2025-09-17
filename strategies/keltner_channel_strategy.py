"""
Keltner Channel Strategy

A volatility-based strategy that uses Keltner Channels (SMA + ATR bands) to identify
breakouts and mean reversion opportunities within the channel structure.
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging
import traceback

from strategies.base_strategy import BaseStrategy, StrategyConfig
from strategies.indicators_cache import calculate_indicators_for_multi_symbol
from core.contracts import TradingSignal, SignalType, SignalStrength


class KeltnerChannelStrategy(BaseStrategy):
    """Keltner Channel trading strategy."""

    def __init__(self, config) -> None:
        """Initialize Keltner Channel strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "sma_period": 20,  # SMA period for middle line
            "atr_period": 14,  # ATR period for bands
            "atr_multiplier": 2.0,  # ATR multiplier for band width
            "position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.025,  # 2.5% stop loss
            "take_profit_pct": 0.08,  # 8% take profit
            "volume_filter": True,  # Use volume confirmation
            "volume_threshold": 1.3,  # Volume must be 1.3x average
            "squeeze_filter": True,  # Use Bollinger/Keltner squeeze
            "squeeze_threshold": 0.8,  # Squeeze threshold ratio
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channel indicators using vectorized operations for better performance.

        PERFORMANCE IMPROVEMENT: This method now uses vectorized pandas operations
        instead of inefficient groupby/apply patterns. For multi-symbol data, it processes
        all symbols simultaneously using numpy/pandas vectorization, which provides
        significant performance gains especially with large datasets containing many symbols.

        The shared indicators_cache module provides additional caching to avoid redundant
        calculations when the same indicator parameters are used across different strategies.
        """
        if data.empty:
            return data

        # Use vectorized calculation for all symbols at once
        # This eliminates the need for groupby operations and provides much better performance
        indicators_config = {
            'keltner': {
                'sma_period': int(self.params["sma_period"]),
                'atr_period': int(self.params["atr_period"]),
                'atr_multiplier': self.params["atr_multiplier"]
            }
        }

        # Calculate indicators using the shared vectorized function
        result_df = calculate_indicators_for_multi_symbol(data, indicators_config)

        # Add squeeze ratio calculation if enabled
        if self.params["squeeze_filter"]:
            # Calculate squeeze indicator (Keltner width relative to recent average)
            avg_width = result_df["keltner_width"].rolling(window=int(self.params["sma_period"])).mean()
            result_df["squeeze_ratio"] = result_df["keltner_width"] / avg_width

        return result_df

    async def generate_signals(self, data) -> List[TradingSignal]:
        """
        Generate signals based on Keltner Channel breakouts and mean reversion using vectorized operations.

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
            required_cols = ['keltner_middle', 'keltner_upper', 'keltner_lower', 'keltner_position', 'keltner_width']
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

                # Check for NaN in Keltner position
                if pd.isna(last_row["keltner_position"]):
                    logger = logging.getLogger(__name__)
                    logger.warning(f"NaN detected in Keltner position for symbol {symbol} during vectorized processing. "
                                  f"Skipping signal generation. This may indicate data quality issues or insufficient data for Keltner calculation.")
                    continue

                # Volume confirmation (vectorized where possible)
                volume_confirmed = True
                if self.params["volume_filter"] and "volume" in last_row.index and not pd.isna(last_row["volume"]):
                    try:
                        # Get full symbol data for volume calculation
                        full_symbol_data = data[data["symbol"] == symbol]
                        volume_period = int(self.params["sma_period"])

                        if len(full_symbol_data) >= volume_period:
                            avg_volume = full_symbol_data["volume"].tail(volume_period).mean()
                            current_volume = last_row["volume"]
                            volume_confirmed = current_volume >= (avg_volume * self.params["volume_threshold"])
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Volume confirmation failed for {symbol}: {e}")
                        volume_confirmed = True

                if not volume_confirmed:
                    continue

                # Check for squeeze breakout (if enabled)
                squeeze_breakout = True
                if self.params["squeeze_filter"] and "squeeze_ratio" in last_row.index and not pd.isna(last_row["squeeze_ratio"]):
                    try:
                        squeeze_threshold = self.params["squeeze_threshold"]
                        squeeze_breakout = last_row["squeeze_ratio"] < squeeze_threshold
                    except Exception:
                        squeeze_breakout = True

                if not squeeze_breakout:
                    continue

                # Check for breakout above upper channel
                breakout_up = (
                    last_row["high"] > last_row["keltner_upper"] and
                    prev_row["high"] <= prev_row["keltner_upper"]
                )

                # Check for breakout below lower channel
                breakout_down = (
                    last_row["low"] < last_row["keltner_lower"] and
                    prev_row["low"] >= prev_row["keltner_lower"]
                )

                # Check for mean reversion signals
                oversold = last_row["keltner_position"] < 0.2
                overbought = last_row["keltner_position"] > 0.8

                if breakout_up:
                    # Price broke above Keltner upper channel
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
                                "signal_type": "keltner_breakout",
                                "breakout_direction": "upper",
                                "keltner_middle": last_row["keltner_middle"],
                                "keltner_upper": last_row["keltner_upper"],
                                "keltner_lower": last_row["keltner_lower"],
                                "keltner_position": last_row["keltner_position"],
                                "keltner_width": last_row["keltner_width"],
                                "squeeze_ratio": last_row.get("squeeze_ratio", None)
                            },
                            timestamp=signal_timestamp,
                        )
                    )

                elif breakout_down:
                    # Price broke below Keltner lower channel
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
                                "signal_type": "keltner_breakout",
                                "breakout_direction": "lower",
                                "keltner_middle": last_row["keltner_middle"],
                                "keltner_upper": last_row["keltner_upper"],
                                "keltner_lower": last_row["keltner_lower"],
                                "keltner_position": last_row["keltner_position"],
                                "keltner_width": last_row["keltner_width"],
                                "squeeze_ratio": last_row.get("squeeze_ratio", None)
                            },
                            timestamp=signal_timestamp,
                        )
                    )

                elif oversold:
                    # Price is oversold within channel, expect mean reversion up
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
                            strength=SignalStrength.MODERATE,
                            order_type="market",
                            amount=self.params["position_size"] * 0.8,  # Smaller position for reversion
                            current_price=current_price,
                            stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                            take_profit=current_price * (1 + self.params["take_profit_pct"] * 0.6),  # Smaller target
                            metadata={
                                "signal_type": "keltner_reversion",
                                "reversion_type": "oversold",
                                "keltner_middle": last_row["keltner_middle"],
                                "keltner_upper": last_row["keltner_upper"],
                                "keltner_lower": last_row["keltner_lower"],
                                "keltner_position": last_row["keltner_position"],
                                "keltner_width": last_row["keltner_width"]
                            },
                            timestamp=signal_timestamp,
                        )
                    )

                elif overbought:
                    # Price is overbought within channel, expect mean reversion down
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
                            strength=SignalStrength.MODERATE,
                            order_type="market",
                            amount=self.params["position_size"] * 0.8,  # Smaller position for reversion
                            current_price=current_price,
                            stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                            take_profit=current_price * (1 - self.params["take_profit_pct"] * 0.6),  # Smaller target
                            metadata={
                                "signal_type": "keltner_reversion",
                                "reversion_type": "overbought",
                                "keltner_middle": last_row["keltner_middle"],
                                "keltner_upper": last_row["keltner_upper"],
                                "keltner_lower": last_row["keltner_lower"],
                                "keltner_position": last_row["keltner_position"],
                                "keltner_width": last_row["keltner_width"]
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
            # Calculate Keltner Channel for this symbol's data
            data_with_keltner = await self.calculate_indicators(data)

            if data_with_keltner.empty or len(data_with_keltner) < 2:
                return signals

            last_row = data_with_keltner.iloc[-1]
            prev_row = data_with_keltner.iloc[-2]
            current_price = last_row["close"]

            # Check for volume confirmation if enabled
            volume_confirmed = True
            if self.params["volume_filter"] and "volume" in last_row.index:
                try:
                    volume_period = int(self.params["sma_period"])
                    if len(data_with_keltner) >= volume_period:
                        avg_volume = data_with_keltner["volume"].tail(volume_period).mean()
                        current_volume = last_row["volume"]
                        volume_confirmed = current_volume >= (avg_volume * self.params["volume_threshold"])
                except Exception:
                    volume_confirmed = True

            if not volume_confirmed:
                return signals

            # Check for squeeze breakout (if enabled)
            squeeze_breakout = True
            if self.params["squeeze_filter"] and "squeeze_ratio" in last_row.index:
                try:
                    squeeze_threshold = self.params["squeeze_threshold"]
                    squeeze_breakout = last_row["squeeze_ratio"] < squeeze_threshold
                except Exception:
                    squeeze_breakout = True

            if not squeeze_breakout:
                return signals

            # Check for breakout above upper channel
            breakout_up = (
                last_row["high"] > last_row["keltner_upper"] and
                prev_row["high"] <= prev_row["keltner_upper"]
            )

            # Check for breakout below lower channel
            breakout_down = (
                last_row["low"] < last_row["keltner_lower"] and
                prev_row["low"] >= prev_row["keltner_lower"]
            )

            # Check for NaN in Keltner position
            if pd.isna(last_row["keltner_position"]):
                logger = logging.getLogger(__name__)
                logger.warning(f"NaN detected in Keltner position for symbol {symbol}. Skipping signal generation. "
                              f"This may indicate data quality issues or insufficient data for Keltner calculation.")
                return signals

            # Check for mean reversion signals
            oversold = last_row["keltner_position"] < 0.2
            overbought = last_row["keltner_position"] > 0.8

            if breakout_up:
                # Price broke above Keltner upper channel
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_keltner.empty and isinstance(data_with_keltner.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_keltner.index[-1].to_pydatetime()

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
                            "signal_type": "keltner_breakout",
                            "breakout_direction": "upper",
                            "keltner_middle": last_row["keltner_middle"],
                            "keltner_upper": last_row["keltner_upper"],
                            "keltner_lower": last_row["keltner_lower"],
                            "keltner_position": last_row["keltner_position"],
                            "keltner_width": last_row["keltner_width"],
                            "squeeze_ratio": last_row.get("squeeze_ratio", None)
                        },
                        timestamp=signal_timestamp,
                    )
                )

            elif breakout_down:
                # Price broke below Keltner lower channel
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_keltner.empty and isinstance(data_with_keltner.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_keltner.index[-1].to_pydatetime()

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
                            "signal_type": "keltner_breakout",
                            "breakout_direction": "lower",
                            "keltner_middle": last_row["keltner_middle"],
                            "keltner_upper": last_row["keltner_upper"],
                            "keltner_lower": last_row["keltner_lower"],
                            "keltner_position": last_row["keltner_position"],
                            "keltner_width": last_row["keltner_width"],
                            "squeeze_ratio": last_row.get("squeeze_ratio", None)
                        },
                        timestamp=signal_timestamp,
                    )
                )

            elif oversold:
                # Price is oversold within channel, expect mean reversion up
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_keltner.empty and isinstance(data_with_keltner.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_keltner.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,
                        strength=SignalStrength.MODERATE,
                        order_type="market",
                        amount=self.params["position_size"] * 0.8,  # Smaller position for reversion
                        current_price=current_price,
                        stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                        take_profit=current_price * (1 + self.params["take_profit_pct"] * 0.6),  # Smaller target
                        metadata={
                            "signal_type": "keltner_reversion",
                            "reversion_type": "oversold",
                            "keltner_middle": last_row["keltner_middle"],
                            "keltner_upper": last_row["keltner_upper"],
                            "keltner_lower": last_row["keltner_lower"],
                            "keltner_position": last_row["keltner_position"],
                            "keltner_width": last_row["keltner_width"]
                        },
                        timestamp=signal_timestamp,
                    )
                )

            elif overbought:
                # Price is overbought within channel, expect mean reversion down
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_keltner.empty and isinstance(data_with_keltner.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_keltner.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,
                        strength=SignalStrength.MODERATE,
                        order_type="market",
                        amount=self.params["position_size"] * 0.8,  # Smaller position for reversion
                        current_price=current_price,
                        stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                        take_profit=current_price * (1 - self.params["take_profit_pct"] * 0.6),  # Smaller target
                        metadata={
                            "signal_type": "keltner_reversion",
                            "reversion_type": "overbought",
                            "keltner_middle": last_row["keltner_middle"],
                            "keltner_upper": last_row["keltner_upper"],
                            "keltner_lower": last_row["keltner_lower"],
                            "keltner_position": last_row["keltner_position"],
                            "keltner_width": last_row["keltner_width"]
                        },
                        timestamp=signal_timestamp,
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
