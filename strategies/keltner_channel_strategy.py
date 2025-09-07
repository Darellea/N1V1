"""
Keltner Channel Strategy

A volatility-based strategy that uses Keltner Channels (SMA + ATR bands) to identify
breakouts and mean reversion opportunities within the channel structure.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
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
        """Calculate Keltner Channel indicators for each symbol."""
        sma_period = int(self.params["sma_period"])
        atr_period = int(self.params["atr_period"])
        atr_multiplier = self.params["atr_multiplier"]

        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_keltner(group):
                # Calculate middle line (SMA)
                group["keltner_middle"] = group["close"].rolling(window=sma_period).mean()

                # Calculate ATR
                high_low = group["high"] - group["low"]
                high_close = (group["high"] - group["close"].shift()).abs()
                low_close = (group["low"] - group["close"].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=atr_period).mean()

                # Calculate Keltner Channel bands
                group["keltner_upper"] = group["keltner_middle"] + (atr * atr_multiplier)
                group["keltner_lower"] = group["keltner_middle"] - (atr * atr_multiplier)

                # Calculate position within channel
                group["keltner_position"] = (
                    (group["close"] - group["keltner_lower"]) /
                    (group["keltner_upper"] - group["keltner_lower"])
                )

                # Calculate channel width
                group["keltner_width"] = (
                    (group["keltner_upper"] - group["keltner_lower"]) /
                    group["keltner_middle"]
                )

                # Calculate squeeze indicator (Keltner width relative to recent average)
                if self.params["squeeze_filter"]:
                    avg_width = group["keltner_width"].rolling(window=sma_period).mean()
                    group["squeeze_ratio"] = group["keltner_width"] / avg_width

                return group

            data = grouped.apply(calculate_keltner).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            data = data.copy()

            # Calculate middle line (SMA)
            data["keltner_middle"] = data["close"].rolling(window=sma_period).mean()

            # Calculate ATR
            high_low = data["high"] - data["low"]
            high_close = (data["high"] - data["close"].shift()).abs()
            low_close = (data["low"] - data["close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=atr_period).mean()

            # Calculate Keltner Channel bands
            data["keltner_upper"] = data["keltner_middle"] + (atr * atr_multiplier)
            data["keltner_lower"] = data["keltner_middle"] - (atr * atr_multiplier)

            # Calculate position within channel
            data["keltner_position"] = (
                (data["close"] - data["keltner_lower"]) /
                (data["keltner_upper"] - data["keltner_lower"])
            )

            # Calculate channel width
            data["keltner_width"] = (
                (data["keltner_upper"] - data["keltner_lower"]) /
                data["keltner_middle"]
            )

            # Calculate squeeze indicator
            if self.params["squeeze_filter"]:
                avg_width = data["keltner_width"].rolling(window=sma_period).mean()
                data["squeeze_ratio"] = data["keltner_width"] / avg_width

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on Keltner Channel breakouts and mean reversion."""
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

            # Check for mean reversion signals
            oversold = (
                last_row["keltner_position"] < 0.2 and
                not pd.isna(last_row["keltner_position"])
            )

            overbought = (
                last_row["keltner_position"] > 0.8 and
                not pd.isna(last_row["keltner_position"])
            )

            if breakout_up:
                # Price broke above Keltner upper channel
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                self.last_signal_time = pd.Timestamp.now()

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
                    )
                )

            elif breakout_down:
                # Price broke below Keltner lower channel
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                self.last_signal_time = pd.Timestamp.now()

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
                    )
                )

            elif oversold:
                # Price is oversold within channel, expect mean reversion up
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                self.last_signal_time = pd.Timestamp.now()

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
                    )
                )

            elif overbought:
                # Price is overbought within channel, expect mean reversion down
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                self.last_signal_time = pd.Timestamp.now()

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
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
