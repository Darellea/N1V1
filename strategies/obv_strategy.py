"""
On-Balance Volume (OBV) Strategy

A volume-based strategy that uses On-Balance Volume to identify accumulation/distribution
patterns and generate signals when volume confirms price movements.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength


class OBVStrategy(BaseStrategy):
    """On-Balance Volume trading strategy."""

    def __init__(self, config) -> None:
        """Initialize OBV strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "obv_signal_period": 10,  # Period for OBV signal line
            "volume_sma_period": 20,  # Period for volume SMA
            "divergence_threshold": 0.02,  # Minimum divergence percentage
            "position_size": 0.09,  # 9% of portfolio
            "stop_loss_pct": 0.03,  # 3% stop loss
            "take_profit_pct": 0.07,  # 7% take profit
            "confirmation_period": 3,  # Bars for signal confirmation
            "trend_filter": True,  # Use trend direction filter
            "trend_period": 20,  # Period for trend calculation
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate OBV indicators for each symbol."""
        signal_period = int(self.params["obv_signal_period"])
        volume_sma_period = int(self.params["volume_sma_period"])
        trend_period = int(self.params["trend_period"])

        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_obv_indicators(group):
                # Calculate OBV
                obv = pd.Series([0] * len(group), index=group.index, dtype=float)
                obv.iloc[0] = group["volume"].iloc[0]

                for i in range(1, len(group)):
                    if group["close"].iloc[i] > group["close"].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + group["volume"].iloc[i]
                    elif group["close"].iloc[i] < group["close"].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - group["volume"].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]

                group["obv"] = obv

                # Calculate OBV signal line (SMA of OBV)
                group["obv_signal"] = group["obv"].rolling(window=signal_period).mean()

                # Calculate OBV momentum
                group["obv_momentum"] = group["obv"].diff()

                # Calculate volume SMA for confirmation
                group["volume_sma"] = group["volume"].rolling(window=volume_sma_period).mean()

                # Calculate trend direction if enabled
                if self.params["trend_filter"]:
                    group["price_trend"] = group["close"].rolling(window=trend_period).mean()

                return group

            data = grouped.apply(calculate_obv_indicators).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            data = data.copy()

            # Calculate OBV
            obv = pd.Series([0] * len(data), index=data.index, dtype=float)
            obv.iloc[0] = data["volume"].iloc[0]

            for i in range(1, len(data)):
                if data["close"].iloc[i] > data["close"].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data["volume"].iloc[i]
                elif data["close"].iloc[i] < data["close"].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data["volume"].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            data["obv"] = obv

            # Calculate OBV signal line
            data["obv_signal"] = data["obv"].rolling(window=signal_period).mean()

            # Calculate OBV momentum
            data["obv_momentum"] = data["obv"].diff()

            # Calculate volume SMA
            data["volume_sma"] = data["volume"].rolling(window=volume_sma_period).mean()

            # Calculate trend direction
            if self.params["trend_filter"]:
                data["price_trend"] = data["close"].rolling(window=trend_period).mean()

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on OBV analysis."""
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
            # Calculate OBV indicators for this symbol's data
            data_with_obv = await self.calculate_indicators(data)

            if data_with_obv.empty or len(data_with_obv) < 2:
                return signals

            last_row = data_with_obv.iloc[-1]
            prev_row = data_with_obv.iloc[-2]
            current_price = last_row["close"]

            # Check for volume confirmation (current volume above average)
            volume_confirmed = True
            if "volume_sma" in last_row.index and not pd.isna(last_row["volume_sma"]):
                volume_confirmed = last_row["volume"] >= last_row["volume_sma"]

            if not volume_confirmed:
                return signals

            # Check for trend confirmation if enabled
            trend_confirmed = True
            if self.params["trend_filter"] and "price_trend" in last_row.index:
                try:
                    # For long signals, prefer uptrending or neutral price
                    # For short signals, prefer downtrending or neutral price
                    trend_confirmed = True  # Allow all trends for now
                except Exception:
                    trend_confirmed = True

            if not trend_confirmed:
                return signals

            # Check for OBV bullish signals
            obv_bullish_crossover = (
                prev_row["obv"] <= prev_row["obv_signal"] and
                last_row["obv"] > last_row["obv_signal"] and
                not pd.isna(last_row["obv"]) and
                not pd.isna(last_row["obv_signal"])
            )

            obv_momentum_positive = (
                last_row["obv_momentum"] > 0 and
                not pd.isna(last_row["obv_momentum"])
            )

            # Check for OBV bearish signals
            obv_bearish_crossover = (
                prev_row["obv"] >= prev_row["obv_signal"] and
                last_row["obv"] < last_row["obv_signal"] and
                not pd.isna(last_row["obv"]) and
                not pd.isna(last_row["obv_signal"])
            )

            obv_momentum_negative = (
                last_row["obv_momentum"] < 0 and
                not pd.isna(last_row["obv_momentum"])
            )

            # Check for confirmation over multiple bars
            confirmation_period = int(self.params["confirmation_period"])
            if len(data_with_obv) >= confirmation_period:
                recent_obv = data_with_obv["obv"].tail(confirmation_period)
                recent_signal = data_with_obv["obv_signal"].tail(confirmation_period)

                # Bullish confirmation: OBV above signal for most recent bars
                bullish_confirmation = (
                    (recent_obv > recent_signal).sum() >= confirmation_period * 0.7
                )

                # Bearish confirmation: OBV below signal for most recent bars
                bearish_confirmation = (
                    (recent_obv < recent_signal).sum() >= confirmation_period * 0.7
                )
            else:
                bullish_confirmation = obv_bullish_crossover
                bearish_confirmation = obv_bearish_crossover

            if (obv_bullish_crossover or obv_momentum_positive) and bullish_confirmation:
                # OBV shows accumulation/bullish divergence
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
                            "signal_type": "obv_accumulation",
                            "obv": last_row["obv"],
                            "obv_signal": last_row["obv_signal"],
                            "obv_momentum": last_row["obv_momentum"],
                            "volume_confirmed": volume_confirmed,
                            "crossover": obv_bullish_crossover,
                            "confirmation_bars": confirmation_period
                        },
                    )
                )

            elif (obv_bearish_crossover or obv_momentum_negative) and bearish_confirmation:
                # OBV shows distribution/bearish divergence
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
                            "signal_type": "obv_distribution",
                            "obv": last_row["obv"],
                            "obv_signal": last_row["obv_signal"],
                            "obv_momentum": last_row["obv_momentum"],
                            "volume_confirmed": volume_confirmed,
                            "crossover": obv_bearish_crossover,
                            "confirmation_bars": confirmation_period
                        },
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
