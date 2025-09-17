"""
ATR Breakout Strategy

A volatility-based strategy that uses Average True Range (ATR) to identify
periods of increasing volatility and generates breakout signals when price
moves beyond ATR-based thresholds.
"""

# Centralized configuration constants for robustness and maintainability
ATR_PERIOD = 14  # ATR calculation period
BREAKOUT_MULTIPLIER = 2.0  # ATR multiplier for breakout threshold
POSITION_SIZE = 0.12  # 12% of portfolio (higher for breakouts)
STOP_LOSS_PCT = 0.03  # 3% stop loss
TAKE_PROFIT_PCT = 0.12  # 12% take profit
VOLUME_FILTER = True  # Use volume confirmation
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x average
MIN_ATR = 0.005  # Minimum ATR value (0.5%)
TREND_FILTER = True  # Use trend direction filter
TREND_PERIOD = 20  # Period for trend calculation

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

from strategies.base_strategy import BaseStrategy, StrategyConfig, SignalGenerationMixin
from core.contracts import TradingSignal, SignalType, SignalStrength


class ATRBreakoutStrategy(BaseStrategy, SignalGenerationMixin):
    """ATR-based breakout trading strategy."""

    def __init__(self, config: Union[StrategyConfig, Dict[str, Any]]) -> None:
        """Initialize ATR breakout strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, Any] = {
            "atr_period": ATR_PERIOD,  # ATR calculation period
            "breakout_multiplier": BREAKOUT_MULTIPLIER,  # ATR multiplier for breakout threshold
            "position_size": POSITION_SIZE,  # 12% of portfolio (higher for breakouts)
            "stop_loss_pct": STOP_LOSS_PCT,  # 3% stop loss
            "take_profit_pct": TAKE_PROFIT_PCT,  # 12% take profit
            "volume_filter": VOLUME_FILTER,  # Use volume confirmation
            "volume_threshold": VOLUME_THRESHOLD,  # Volume must be 1.5x average
            "min_atr": MIN_ATR,  # Minimum ATR value (0.5%)
            "trend_filter": TREND_FILTER,  # Use trend direction filter
            "trend_period": TREND_PERIOD,  # Period for trend calculation
        }
        self.params = self._merge_params(self.default_params, config)

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR and breakout indicators for each symbol."""
        # Input validation: Check for required columns
        required_columns = ['open', 'high', 'low', 'close']
        if self.params["volume_filter"]:
            required_columns.append('volume')
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in market data: {missing_columns}. "
                           f"Expected columns: {required_columns}")

        atr_period = int(self.params["atr_period"])
        trend_period = int(self.params["trend_period"])

        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_atr_indicators(group):
                # Calculate ATR
                high_low = group["high"] - group["low"]
                high_close = (group["high"] - group["close"].shift()).abs()
                low_close = (group["low"] - group["close"].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                group["atr"] = true_range.rolling(window=atr_period).mean()

                # Calculate breakout thresholds
                multiplier = self.params["breakout_multiplier"]
                group["breakout_upper"] = group["close"] + (group["atr"] * multiplier)
                group["breakout_lower"] = group["close"] - (group["atr"] * multiplier)

                # Calculate trend direction (simple moving average slope)
                if self.params["trend_filter"]:
                    sma = group["close"].rolling(window=trend_period).mean()
                    group["trend_slope"] = sma.diff()

                return group

            data = grouped.apply(calculate_atr_indicators).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            data = data.copy()

            # Calculate ATR
            high_low = data["high"] - data["low"]
            high_close = (data["high"] - data["close"].shift()).abs()
            low_close = (data["low"] - data["close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data["atr"] = true_range.rolling(window=atr_period).mean()

            # Calculate breakout thresholds
            multiplier = self.params["breakout_multiplier"]
            data["breakout_upper"] = data["close"] + (data["atr"] * multiplier)
            data["breakout_lower"] = data["close"] - (data["atr"] * multiplier)

            # Calculate trend direction
            if self.params["trend_filter"]:
                sma = data["close"].rolling(window=trend_period).mean()
                data["trend_slope"] = sma.diff()

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on ATR breakouts."""
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
            # Calculate ATR indicators for this symbol's data
            data_with_atr = await self.calculate_indicators(data)

            if data_with_atr.empty or len(data_with_atr) < 2:
                return signals

            last_row = data_with_atr.iloc[-1]
            prev_row = data_with_atr.iloc[-2]
            current_price = last_row["close"]

            # Check for volume confirmation if enabled
            volume_confirmed = True
            if self.params["volume_filter"] and "volume" in last_row.index:
                try:
                    volume_period = int(self.params["atr_period"])
                    if len(data_with_atr) >= volume_period:
                        avg_volume = data_with_atr["volume"].tail(volume_period).mean()
                        current_volume = last_row["volume"]
                        volume_confirmed = current_volume >= (avg_volume * self.params["volume_threshold"])
                except Exception:
                    volume_confirmed = True

            if not volume_confirmed:
                return signals

            # Check ATR minimum threshold
            min_atr = self.params["min_atr"]
            if last_row["atr"] < min_atr:
                return signals  # ATR too low, avoid low volatility breakouts

            # Check for breakout above upper threshold
            breakout_up = (
                last_row["high"] > last_row["breakout_upper"] and
                prev_row["high"] <= prev_row["breakout_upper"]
            )

            # Check for breakout below lower threshold
            breakout_down = (
                last_row["low"] < last_row["breakout_lower"] and
                prev_row["low"] >= prev_row["breakout_lower"]
            )

            # Check for trend filter if enabled
            trend_confirmed = True
            if self.params["trend_filter"] and "trend_slope" in last_row.index:
                try:
                    trend_slope = last_row["trend_slope"]
                    if not pd.isna(trend_slope):
                        # For long signals, prefer uptrend or neutral (trend_slope >= 0)
                        # For short signals, prefer downtrend or neutral (trend_slope <= 0)
                        if breakout_up and trend_slope < 0:
                            trend_confirmed = False  # Strong downtrend against long breakout
                        elif breakout_down and trend_slope > 0:
                            trend_confirmed = False  # Strong uptrend against short breakout
                        # Allow neutral trends (trend_slope == 0) for both directions
                except Exception:
                    trend_confirmed = True

            if not trend_confirmed:
                return signals

            if breakout_up:
                # Price broke above ATR upper threshold
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_atr.empty and isinstance(data_with_atr.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_atr.index[-1].to_pydatetime()

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
                            "breakout_type": "atr_upper",
                            "atr": last_row["atr"],
                            "breakout_upper": last_row["breakout_upper"],
                            "breakout_lower": last_row["breakout_lower"],
                            "breakout_multiplier": self.params["breakout_multiplier"],
                            "volatility_confirmed": volume_confirmed
                        },
                        timestamp=signal_timestamp,
                    )
                )

            elif breakout_down:
                # Price broke below ATR lower threshold
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_atr.empty and isinstance(data_with_atr.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_atr.index[-1].to_pydatetime()

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
                            "breakout_type": "atr_lower",
                            "atr": last_row["atr"],
                            "breakout_upper": last_row["breakout_upper"],
                            "breakout_lower": last_row["breakout_lower"],
                            "breakout_multiplier": self.params["breakout_multiplier"],
                            "volatility_confirmed": volume_confirmed
                        },
                        timestamp=signal_timestamp,
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
