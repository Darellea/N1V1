# strategies/ema_cross_strategy.py
import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.signal_router import TradingSignal, SignalType, SignalStrength


class EMACrossStrategy(BaseStrategy):
    """EMA Crossover trading strategy."""

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize EMA crossover strategy.

        Args:
            config: StrategyConfig instance with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "fast_ema": 9,
            "slow_ema": 21,
            "position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.03,  # 3%
            "take_profit_pct": 0.06,  # 6%
        }
        self.params: Dict[str, float] = {**self.default_params, **(config.params or {})}

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMAs for each symbol."""
        grouped = data.groupby("symbol")

        def calculate_emas(group):
            group["fast_ema"] = (
                group["close"].ewm(span=self.params["fast_ema"], adjust=False).mean()
            )
            group["slow_ema"] = (
                group["close"].ewm(span=self.params["slow_ema"], adjust=False).mean()
            )
            return group

        data = grouped.apply(calculate_emas).reset_index(level=0, drop=True)
        return data

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on EMA crossovers."""
        signals = []
        grouped = data.groupby("symbol")

        for symbol, group in grouped:
            if len(group) < 2:  # Need at least 2 data points to check crossover
                continue

            last_row = group.iloc[-1]
            prev_row = group.iloc[-2]
            current_price = last_row["close"]

            # Check for bullish crossover (fast EMA crosses above slow EMA)
            if (
                prev_row["fast_ema"] <= prev_row["slow_ema"]
                and last_row["fast_ema"] > last_row["slow_ema"]
            ):
                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,  # Changed from BUY to ENTRY_LONG
                        strength=SignalStrength.MODERATE,
                        order_type="market",
                        amount=self.params["position_size"],
                        current_price=current_price,
                        stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                        take_profit=current_price
                        * (1 + self.params["take_profit_pct"]),
                        metadata={
                            "fast_ema": last_row["fast_ema"],
                            "slow_ema": last_row["slow_ema"],
                        },
                    )
                )

            # Check for bearish crossover (fast EMA crosses below slow EMA)
            elif (
                prev_row["fast_ema"] >= prev_row["slow_ema"]
                and last_row["fast_ema"] < last_row["slow_ema"]
            ):
                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,  # Changed from SELL to ENTRY_SHORT
                        strength=SignalStrength.MODERATE,
                        order_type="market",
                        amount=self.params["position_size"],
                        current_price=current_price,
                        stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                        take_profit=current_price
                        * (1 - self.params["take_profit_pct"]),
                        metadata={
                            "fast_ema": last_row["fast_ema"],
                            "slow_ema": last_row["slow_ema"],
                        },
                    )
                )

        return signals
