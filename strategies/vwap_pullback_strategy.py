"""
VWAP Pullback Strategy

A volume-based strategy that uses Volume Weighted Average Price (VWAP) to identify
key support/resistance levels and generates signals on pullbacks to VWAP with
volume confirmation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength


class VWAPPullbackStrategy(BaseStrategy):
    """VWAP pullback trading strategy."""

    def __init__(self, config) -> None:
        """Initialize VWAP pullback strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "vwap_period": 20,  # Period for VWAP calculation
            "pullback_threshold": 0.01,  # Maximum pullback percentage from VWAP
            "volume_multiplier": 1.3,  # Volume must be 1.3x average for breakout
            "position_size": 0.08,  # 8% of portfolio
            "stop_loss_pct": 0.025,  # 2.5% stop loss
            "take_profit_pct": 0.06,  # 6% take profit
            "confirmation_bars": 2,  # Bars for signal confirmation
            "trend_filter": True,  # Use trend direction filter
            "trend_period": 10,  # Period for trend calculation
            "min_volume_period": 10,  # Minimum period for volume analysis
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP and related indicators for each symbol."""
        vwap_period = int(self.params["vwap_period"])
        trend_period = int(self.params["trend_period"])
        min_volume_period = int(self.params["min_volume_period"])

        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_vwap_indicators(group):
                # Calculate VWAP (Volume Weighted Average Price)
                group["vwap"] = (
                    (group["close"] * group["volume"]).cumsum() /
                    group["volume"].cumsum()
                )

                # Calculate VWAP deviation
                group["vwap_deviation"] = (
                    (group["close"] - group["vwap"]) / group["vwap"]
                )

                # Calculate volume moving average for confirmation
                group["volume_ma"] = group["volume"].rolling(window=min_volume_period).mean()

                # Calculate trend direction if enabled
                if self.params["trend_filter"]:
                    group["price_trend"] = group["close"].rolling(window=trend_period).mean()
                    group["trend_slope"] = group["price_trend"].diff()

                # Calculate VWAP slope (trend of VWAP itself)
                group["vwap_slope"] = group["vwap"].diff()

                return group

            data = grouped.apply(calculate_vwap_indicators).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            data = data.copy()

            # Calculate VWAP
            data["vwap"] = (
                (data["close"] * data["volume"]).cumsum() /
                data["volume"].cumsum()
            )

            # Calculate VWAP deviation
            data["vwap_deviation"] = (
                (data["close"] - data["vwap"]) / data["vwap"]
            )

            # Calculate volume moving average
            data["volume_ma"] = data["volume"].rolling(window=min_volume_period).mean()

            # Calculate trend direction
            if self.params["trend_filter"]:
                data["price_trend"] = data["close"].rolling(window=trend_period).mean()
                data["trend_slope"] = data["price_trend"].diff()

            # Calculate VWAP slope
            data["vwap_slope"] = data["vwap"].diff()

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on VWAP pullbacks."""
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
            # Calculate VWAP indicators for this symbol's data
            data_with_vwap = await self.calculate_indicators(data)

            if data_with_vwap.empty or len(data_with_vwap) < 2:
                return signals

            last_row = data_with_vwap.iloc[-1]
            prev_row = data_with_vwap.iloc[-2]
            current_price = last_row["close"]

            # Check for volume confirmation
            volume_confirmed = True
            if "volume_ma" in last_row.index and not pd.isna(last_row["volume_ma"]):
                volume_confirmed = last_row["volume"] >= (
                    last_row["volume_ma"] * self.params["volume_multiplier"]
                )

            if not volume_confirmed:
                return signals

            # Check for trend confirmation if enabled
            trend_confirmed = True
            if self.params["trend_filter"] and "trend_slope" in last_row.index:
                try:
                    trend_slope = last_row["trend_slope"]
                    vwap_slope = last_row["vwap_slope"]

                    # For long signals, prefer uptrending price and VWAP
                    # For short signals, prefer downtrending price and VWAP
                    if not pd.isna(trend_slope) and not pd.isna(vwap_slope):
                        # Allow signals if both price and VWAP are moving in same direction
                        trend_confirmed = (
                            (trend_slope > 0 and vwap_slope > 0) or
                            (trend_slope < 0 and vwap_slope < 0) or
                            abs(trend_slope) < 0.001  # Neutral trend allowed
                        )
                    else:
                        trend_confirmed = True
                except Exception:
                    trend_confirmed = True

            if not trend_confirmed:
                return signals

            # Check for pullback to VWAP (price close to VWAP)
            pullback_threshold = self.params["pullback_threshold"]
            vwap_deviation = abs(last_row["vwap_deviation"])

            near_vwap = (
                vwap_deviation <= pullback_threshold and
                not pd.isna(last_row["vwap_deviation"])
            )

            if not near_vwap:
                return signals

            # Check for confirmation over multiple bars
            confirmation_bars = int(self.params["confirmation_bars"])
            if len(data_with_vwap) >= confirmation_bars:
                recent_prices = data_with_vwap["close"].tail(confirmation_bars)
                recent_vwap = data_with_vwap["vwap"].tail(confirmation_bars)

                # Check if price has been consistently near VWAP
                deviations = abs((recent_prices - recent_vwap) / recent_vwap)
                avg_deviation = deviations.mean()

                confirmed_pullback = avg_deviation <= pullback_threshold
            else:
                confirmed_pullback = near_vwap

            if not confirmed_pullback:
                return signals

            # Determine signal direction based on recent price action and VWAP position
            price_above_vwap = last_row["close"] > last_row["vwap"]
            prev_price_above_vwap = prev_row["close"] > prev_row["vwap"]

            # Bullish signal: price pulls back to VWAP and shows upward momentum
            bullish_signal = (
                price_above_vwap and
                (last_row["close"] > prev_row["close"] or prev_price_above_vwap)
            )

            # Bearish signal: price pulls back to VWAP and shows downward momentum
            bearish_signal = (
                not price_above_vwap and
                (last_row["close"] < prev_row["close"] or not prev_price_above_vwap)
            )

            if bullish_signal:
                # Price is above VWAP and showing upward momentum after pullback
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                self.last_signal_time = pd.Timestamp.now()

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
                            "signal_type": "vwap_pullback_long",
                            "vwap": last_row["vwap"],
                            "vwap_deviation": last_row["vwap_deviation"],
                            "pullback_threshold": pullback_threshold,
                            "volume_confirmed": volume_confirmed,
                            "trend_confirmed": trend_confirmed,
                            "confirmation_bars": confirmation_bars,
                            "price_above_vwap": price_above_vwap
                        },
                    )
                )

            elif bearish_signal:
                # Price is below VWAP and showing downward momentum after pullback
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                self.last_signal_time = pd.Timestamp.now()

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
                            "signal_type": "vwap_pullback_short",
                            "vwap": last_row["vwap"],
                            "vwap_deviation": last_row["vwap_deviation"],
                            "pullback_threshold": pullback_threshold,
                            "volume_confirmed": volume_confirmed,
                            "trend_confirmed": trend_confirmed,
                            "confirmation_bars": confirmation_bars,
                            "price_above_vwap": price_above_vwap
                        },
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
