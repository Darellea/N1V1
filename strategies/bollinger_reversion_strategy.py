"""
Bollinger Bands Mean Reversion Strategy

A range-based strategy that trades when price touches or moves beyond the
Bollinger Bands, expecting mean reversion back to the middle band.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength


class BollingerReversionStrategy(BaseStrategy):
    """Bollinger Bands mean reversion trading strategy."""

    def __init__(self, config) -> None:
        """Initialize Bollinger reversion strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "period": 20,  # Bollinger Bands period
            "std_dev": 2.0,  # Standard deviation multiplier
            "reversion_threshold": 0.01,  # Minimum distance from band for signal
            "position_size": 0.08,  # 8% of portfolio (smaller for mean reversion)
            "stop_loss_pct": 0.02,  # 2% stop loss
            "take_profit_pct": 0.04,  # 4% take profit (smaller targets)
            "volume_filter": True,  # Use volume confirmation
            "volume_threshold": 1.1,  # Volume must be 1.1x average
            "max_holding_period": 10,  # Maximum bars to hold position
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators for each symbol."""
        period = int(self.params["period"])
        std_dev = self.params["std_dev"]

        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_bollinger(group):
                # Calculate Bollinger Bands
                sma = group["close"].rolling(window=period).mean()
                std = group["close"].rolling(window=period).std()

                group["bb_middle"] = sma
                group["bb_upper"] = sma + (std * std_dev)
                group["bb_lower"] = sma - (std * std_dev)

                # Calculate position within bands
                group["bb_position"] = (group["close"] - group["bb_lower"]) / (group["bb_upper"] - group["bb_lower"])

                # Calculate band width
                group["bb_width"] = (group["bb_upper"] - group["bb_lower"]) / group["bb_middle"]

                return group

            data = grouped.apply(calculate_bollinger).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            data = data.copy()
            sma = data["close"].rolling(window=period).mean()
            std = data["close"].rolling(window=period).std()

            data["bb_middle"] = sma
            data["bb_upper"] = sma + (std * std_dev)
            data["bb_lower"] = sma - (std * std_dev)
            data["bb_position"] = (data["close"] - data["bb_lower"]) / (data["bb_upper"] - data["bb_lower"])
            data["bb_width"] = (data["bb_upper"] - data["bb_lower"]) / data["bb_middle"]

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on Bollinger Band mean reversion."""
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
            # Calculate Bollinger Bands for this symbol's data
            data_with_bb = await self.calculate_indicators(data)

            if data_with_bb.empty or len(data_with_bb) < 2:
                return signals

            last_row = data_with_bb.iloc[-1]
            prev_row = data_with_bb.iloc[-2]
            current_price = last_row["close"]

            # Check for volume confirmation if enabled
            volume_confirmed = True
            if self.params["volume_filter"] and "volume" in last_row.index:
                try:
                    volume_period = int(self.params["period"])
                    if len(data_with_bb) >= volume_period:
                        avg_volume = data_with_bb["volume"].tail(volume_period).mean()
                        current_volume = last_row["volume"]
                        volume_confirmed = current_volume >= (avg_volume * self.params["volume_threshold"])
                except Exception:
                    volume_confirmed = True

            if not volume_confirmed:
                return signals

            # Check for oversold condition (price near lower band)
            oversold_threshold = self.params["reversion_threshold"]
            oversold = (
                last_row["bb_position"] < oversold_threshold and
                not pd.isna(last_row["bb_position"])
            )

            # Check for overbought condition (price near upper band)
            overbought_threshold = 1.0 - self.params["reversion_threshold"]
            overbought = (
                last_row["bb_position"] > overbought_threshold and
                not pd.isna(last_row["bb_position"])
            )

            # Additional filter: ensure bands are not too narrow (avoid whipsaw)
            min_bandwidth = 0.005  # Minimum 0.5% bandwidth
            valid_bandwidth = last_row["bb_width"] > min_bandwidth

            if oversold and valid_bandwidth:
                # Price is oversold, expect mean reversion up
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
                            "reversion_type": "oversold",
                            "bb_position": last_row["bb_position"],
                            "bb_middle": last_row["bb_middle"],
                            "bb_upper": last_row["bb_upper"],
                            "bb_lower": last_row["bb_lower"],
                            "bb_width": last_row["bb_width"]
                        },
                    )
                )

            elif overbought and valid_bandwidth:
                # Price is overbought, expect mean reversion down
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
                            "reversion_type": "overbought",
                            "bb_position": last_row["bb_position"],
                            "bb_middle": last_row["bb_middle"],
                            "bb_upper": last_row["bb_upper"],
                            "bb_lower": last_row["bb_lower"],
                            "bb_width": last_row["bb_width"]
                        },
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
