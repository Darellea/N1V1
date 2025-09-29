# strategies/ema_cross_strategy.py
from typing import Dict, List

import pandas as pd

from core.contracts import SignalStrength, SignalType, TradingSignal
from strategies.base_strategy import BaseStrategy


class EMACrossStrategy(BaseStrategy):
    """EMA Crossover trading strategy."""

    def __init__(self, config) -> None:
        """Initialize EMA crossover strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, float] = {
            "fast_ema": 9,
            "slow_ema": 21,
            "position_size": 0.1,  # 10% of portfolio
            "stop_loss_pct": 0.03,  # 3%
            "take_profit_pct": 0.06,  # 6%
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = (
            config.params if hasattr(config, "params") else config.get("params", {})
        )
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

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

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on EMA crossovers."""
        signals = []

        # Handle different data formats
        if isinstance(data, dict):
            # Data is a dict of {symbol: DataFrame}
            for symbol, df in data.items():
                if df is not None and not df.empty:
                    signals.extend(await self._generate_signals_for_symbol(symbol, df))
        elif hasattr(data, "groupby"):
            # Data is a single DataFrame
            grouped = data.groupby("symbol")
            for symbol, group in grouped:
                signals.extend(await self._generate_signals_for_symbol(symbol, group))
        else:
            # Fallback: try to convert to DataFrame
            import pandas as pd

            try:
                df = pd.DataFrame(data)
                grouped = df.groupby("symbol")
                for symbol, group in grouped:
                    signals.extend(
                        await self._generate_signals_for_symbol(symbol, group)
                    )
            except Exception:
                pass

        return signals

    async def _generate_signals_for_symbol(
        self, symbol: str, data
    ) -> List[TradingSignal]:
        """Generate signals for a specific symbol's data."""
        signals = []

        try:
            # Calculate EMAs for this symbol's data
            data_with_emas = await self.calculate_indicators(data)

            if data_with_emas.empty or len(data_with_emas) < 2:
                return signals

            last_row = data_with_emas.iloc[-1]
            prev_row = data_with_emas.iloc[-2]
            current_price = last_row["close"]

            # Check for bullish crossover (fast EMA crosses above slow EMA)
            if (
                prev_row["fast_ema"] <= prev_row["slow_ema"]
                and last_row["fast_ema"] > last_row["slow_ema"]
            ):
                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_emas.empty and isinstance(
                    data_with_emas.index, pd.DatetimeIndex
                ):
                    signal_timestamp = data_with_emas.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,
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
                        timestamp=signal_timestamp,
                    )
                )

            # Check for bearish crossover (fast EMA crosses below slow EMA)
            elif (
                prev_row["fast_ema"] >= prev_row["slow_ema"]
                and last_row["fast_ema"] < last_row["slow_ema"]
            ):
                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_emas.empty and isinstance(
                    data_with_emas.index, pd.DatetimeIndex
                ):
                    signal_timestamp = data_with_emas.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,
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
                        timestamp=signal_timestamp,
                    )
                )

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
