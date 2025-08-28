# strategies/rsi_strategy.py
import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.signal_router import TradingSignal, SignalType, SignalStrength


class RSIStrategy(BaseStrategy):
    """Relative Strength Index trading strategy."""

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize RSI strategy.

        Args:
            config: StrategyConfig instance with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "rsi_period": 14,
            "overbought": 70,
            "oversold": 30,
            "position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.05,  # 5%
            "take_profit_pct": 0.1,  # 10%
        }
        self.params: Dict[str, float] = {**self.default_params, **(config.params or {})}

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator for each symbol."""
        grouped = data.groupby("symbol")

        def calculate_rsi(group):
            delta = group["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(self.params["rsi_period"]).mean()
            avg_loss = loss.rolling(self.params["rsi_period"]).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        data["rsi"] = grouped.apply(calculate_rsi).reset_index(level=0, drop=True)
        return data

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate signals based on RSI values."""
        signals = []
        grouped = data.groupby("symbol")

        for symbol, group in grouped:
            last_row = group.iloc[-1]
            current_price = last_row["close"]

            if last_row["rsi"] > self.params["overbought"]:
                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,  # Changed from SELL to ENTRY_SHORT
                        strength=SignalStrength.STRONG,
                        order_type="market",
                        amount=self.params["position_size"],
                        current_price=current_price,
                        stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                        take_profit=current_price
                        * (1 - self.params["take_profit_pct"]),
                        metadata={"rsi_value": last_row["rsi"]},
                    )
                )

            elif last_row["rsi"] < self.params["oversold"]:
                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,  # Changed from BUY to ENTRY_LONG
                        strength=SignalStrength.STRONG,
                        order_type="market",
                        amount=self.params["position_size"],
                        current_price=current_price,
                        stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                        take_profit=current_price
                        * (1 + self.params["take_profit_pct"]),
                        metadata={"rsi_value": last_row["rsi"]},
                    )
                )

        return signals
