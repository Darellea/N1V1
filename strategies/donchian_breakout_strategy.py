"""
Donchian Breakout Strategy

A trend-following strategy that uses Donchian channels to identify breakouts
above resistance or below support levels, indicating potential trend continuation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
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
            "channel_period": 20,  # Period for Donchian channel
            "breakout_threshold": 0.001,  # Minimum breakout percentage
            "position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.04,  # 4%
            "take_profit_pct": 0.1,  # 10%
            "volume_filter": True,  # Use volume confirmation
            "volume_multiplier": 1.2,  # Volume must be 1.2x average
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Donchian Channel indicators for each symbol."""
        channel_period = int(self.params["channel_period"])

        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_donchian(group):
                # Calculate Donchian Channel
                group["donchian_high"] = group["high"].rolling(window=channel_period).max()
                group["donchian_low"] = group["low"].rolling(window=channel_period).min()
                group["donchian_mid"] = (group["donchian_high"] + group["donchian_low"]) / 2

                # Calculate channel width
                group["donchian_width"] = group["donchian_high"] - group["donchian_low"]
                group["donchian_width_pct"] = group["donchian_width"] / group["donchian_mid"]

                return group

            data = grouped.apply(calculate_donchian).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            data = data.copy()
            data["donchian_high"] = data["high"].rolling(window=channel_period).max()
            data["donchian_low"] = data["low"].rolling(window=channel_period).min()
            data["donchian_mid"] = (data["donchian_high"] + data["donchian_low"]) / 2
            data["donchian_width"] = data["donchian_high"] - data["donchian_low"]
            data["donchian_width_pct"] = data["donchian_width"] / data["donchian_mid"]

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on Donchian breakouts."""
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
                                "breakout_type": "upper",
                                "donchian_high": last_row["donchian_high"],
                                "donchian_low": last_row["donchian_low"],
                                "breakout_pct": breakout_pct,
                                "channel_width_pct": last_row["donchian_width_pct"]
                            },
                        )
                    )

            elif breakout_down:
                breakout_pct = (last_row["donchian_low"] - last_row["low"]) / last_row["donchian_low"]
                if breakout_pct > breakout_threshold:
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
                                "breakout_type": "lower",
                                "donchian_high": last_row["donchian_high"],
                                "donchian_low": last_row["donchian_low"],
                                "breakout_pct": breakout_pct,
                                "channel_width_pct": last_row["donchian_width_pct"]
                            },
                        )
                    )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
