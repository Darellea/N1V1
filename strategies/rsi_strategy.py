# strategies/rsi_strategy.py
import numpy as np
import pandas as pd
from typing import List, Dict

from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength


class RSIStrategy(BaseStrategy):
    """Relative Strength Index trading strategy."""

    def __init__(self, config) -> None:
        """Initialize RSI strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
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
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

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

    async def generate_signals(self, data) -> List[TradingSignal]:
        """Generate signals based on RSI values."""
        signals = []

        # Handle different data formats
        if isinstance(data, dict):
            # Data is a dict of {symbol: DataFrame}
            for symbol, df in data.items():
                if df is not None and not df.empty:
                    signals.extend(await self._generate_signals_for_symbol(symbol, df))
        elif hasattr(data, 'groupby'):
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
                    signals.extend(await self._generate_signals_for_symbol(symbol, group))
            except Exception:
                pass

        return signals

    async def _generate_signals_for_symbol(self, symbol: str, data) -> List[TradingSignal]:
        """Generate signals for a specific symbol's data."""
        signals = []

        try:
            # Calculate RSI for this symbol's data
            data_with_rsi = await self.calculate_indicators(data)

            if data_with_rsi.empty:
                return signals

            last_row = data_with_rsi.iloc[-1]
            current_price = last_row["close"]

            if last_row["rsi"] > self.params["overbought"]:
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
                        metadata={"rsi_value": last_row["rsi"]},
                    )
                )

            elif last_row["rsi"] < self.params["oversold"]:
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
                        metadata={"rsi_value": last_row["rsi"]},
                    )
                )

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error generating signals for {symbol}: {str(e)}")

        return signals
