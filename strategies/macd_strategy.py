"""
MACD (Moving Average Convergence Divergence) Strategy

A trend-following momentum strategy that uses MACD crossovers and signal line
crossovers to identify trend changes and momentum shifts.
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength


class MACDStrategy(BaseStrategy):
    """MACD trading strategy with signal line crossovers."""

    def __init__(self, config) -> None:
        """Initialize MACD strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.03,  # 3%
            "take_profit_pct": 0.08,  # 8%
            "histogram_threshold": 0.0,  # Minimum histogram value for signals
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicators for each symbol."""
        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_macd(group):
                # Calculate EMAs
                fast_ema = group["close"].ewm(span=self.params["fast_period"], adjust=False).mean()
                slow_ema = group["close"].ewm(span=self.params["slow_period"], adjust=False).mean()

                # Calculate MACD line
                macd_line = fast_ema - slow_ema

                # Calculate signal line
                signal_line = macd_line.ewm(span=self.params["signal_period"], adjust=False).mean()

                # Calculate histogram
                histogram = macd_line - signal_line

                group = group.copy()
                group["macd"] = macd_line
                group["macd_signal"] = signal_line
                group["macd_histogram"] = histogram

                return group

            data = grouped.apply(calculate_macd).reset_index(level=0, drop=True)
        else:
            # Single symbol data
            fast_ema = data["close"].ewm(span=self.params["fast_period"], adjust=False).mean()
            slow_ema = data["close"].ewm(span=self.params["slow_period"], adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=self.params["signal_period"], adjust=False).mean()
            histogram = macd_line - signal_line

            data = data.copy()
            data["macd"] = macd_line
            data["macd_signal"] = signal_line
            data["macd_histogram"] = histogram

        return data

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on MACD crossovers."""
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
            # Calculate MACD for this symbol's data
            data_with_macd = await self.calculate_indicators(data)

            if data_with_macd.empty or len(data_with_macd) < 2:
                return signals

            last_row = data_with_macd.iloc[-1]
            prev_row = data_with_macd.iloc[-2]
            current_price = last_row["close"]

            # Check for MACD bullish crossover (MACD crosses above signal line)
            macd_crossover_bullish = (
                prev_row["macd"] <= prev_row["macd_signal"] and
                last_row["macd"] > last_row["macd_signal"]
            )

            # Check for MACD bearish crossover (MACD crosses below signal line)
            macd_crossover_bearish = (
                prev_row["macd"] >= prev_row["macd_signal"] and
                last_row["macd"] < last_row["macd_signal"]
            )

            # Additional filter: histogram should be significant
            histogram_threshold = self.params["histogram_threshold"]

            if macd_crossover_bullish and abs(last_row["macd_histogram"]) > histogram_threshold:
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
                            "macd": last_row["macd"],
                            "macd_signal": last_row["macd_signal"],
                            "macd_histogram": last_row["macd_histogram"],
                            "crossover_type": "bullish"
                        },
                    )
                )

            elif macd_crossover_bearish and abs(last_row["macd_histogram"]) > histogram_threshold:
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
                            "macd": last_row["macd"],
                            "macd_signal": last_row["macd_signal"],
                            "macd_histogram": last_row["macd_histogram"],
                            "crossover_type": "bearish"
                        },
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
