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
            "volume_period": 10,  # Period for volume confirmation
            "volume_threshold": 1.5,  # Volume must be 1.5x 10-period average
        }
        # Handle both StrategyConfig objects and dict configs
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        self.params: Dict[str, float] = {**self.default_params, **(config_params or {})}

        # Signal tracking for monitoring
        self.signal_counts = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator for each symbol."""
        # Check if data has a symbol column and multiple symbols
        if "symbol" in data.columns and data["symbol"].nunique() > 1:
            grouped = data.groupby("symbol")

            def calculate_rsi(group):
                delta = group["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                avg_gain = gain.ewm(span=self.params["rsi_period"], adjust=False).mean()
                avg_loss = loss.ewm(span=self.params["rsi_period"], adjust=False).mean()

                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            data["rsi"] = grouped.apply(calculate_rsi).reset_index(level=0, drop=True)
        else:
            # Single symbol data - calculate RSI directly
            delta = data["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.ewm(span=self.params["rsi_period"], adjust=False).mean()
            avg_loss = loss.ewm(span=self.params["rsi_period"], adjust=False).mean()

            rs = avg_gain / avg_loss
            data["rsi"] = 100 - (100 / (1 + rs))

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
        elif hasattr(data, 'groupby') and not data.empty and "symbol" in data.columns:
            # Data is a single DataFrame with symbol column
            grouped = data.groupby("symbol")
            for symbol, group in grouped:
                signals.extend(await self._generate_signals_for_symbol(symbol, group))
        elif hasattr(data, 'empty') and data.empty:
            # Empty DataFrame - return empty signals
            return signals
        else:
            # Fallback: try to convert to DataFrame
            import pandas as pd
            try:
                df = pd.DataFrame(data)
                if not df.empty and "symbol" in df.columns:
                    grouped = df.groupby("symbol")
                    for symbol, group in grouped:
                        signals.extend(await self._generate_signals_for_symbol(symbol, group))
            except Exception:
                pass

        return signals

    def _calculate_dynamic_position_size(self, symbol: str) -> float:
        """Calculate dynamic position size based on portfolio allocation or ATR."""
        # For now, use base position size
        # In a full implementation, this would consider:
        # - Portfolio allocation for the symbol
        # - ATR-based volatility adjustment
        # - Account balance and risk management
        return self.params["position_size"]

    async def _generate_signals_for_symbol(self, symbol: str, data) -> List[TradingSignal]:
        """Generate signals for a specific symbol's data."""
        signals = []

        try:
            # Check if RSI is already calculated
            if "rsi" in data.columns and not data["rsi"].isna().all():
                data_with_rsi = data
            else:
                # Calculate RSI for this symbol's data
                data_with_rsi = await self.calculate_indicators(data)

            if data_with_rsi.empty:
                return signals

            last_row = data_with_rsi.iloc[-1]
            current_price = last_row["close"]

            # Only generate signals if RSI is not NaN
            if not pd.isna(last_row["rsi"]):
                # Volume confirmation filter
                volume_confirmed = True
                if "volume" in last_row.index and not pd.isna(last_row["volume"]):
                    try:
                        # Calculate average volume over the specified period
                        volume_period = int(self.params.get("volume_period", 10))
                        if len(data_with_rsi) >= volume_period:
                            avg_volume = data_with_rsi["volume"].tail(volume_period).mean()
                            current_volume = last_row["volume"]
                            volume_threshold = self.params.get("volume_threshold", 1.5)
                            volume_confirmed = current_volume >= (avg_volume * volume_threshold)
                        else:
                            # Not enough data for volume check, proceed
                            volume_confirmed = True
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Volume confirmation failed for {symbol}: {e}")
                        volume_confirmed = True  # Default to allowing signal

                if not volume_confirmed:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Volume confirmation failed for {symbol}: signal skipped")
                    return signals

                if last_row["rsi"] > self.params["overbought"]:
                    # Track signal
                    self.signal_counts["short"] += 1
                    self.signal_counts["total"] += 1
                    self.last_signal_time = pd.Timestamp.now()

                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"RSI SHORT signal for {symbol}: RSI={last_row['rsi']:.2f}, total signals: {self.signal_counts['total']}")

                    signals.append(
                        self.create_signal(
                            symbol=symbol,
                            signal_type=SignalType.ENTRY_SHORT,
                            strength=SignalStrength.STRONG,
                            order_type="market",
                            amount=self._calculate_dynamic_position_size(symbol),
                            current_price=current_price,
                            stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                            take_profit=current_price * (1 - self.params["take_profit_pct"]),
                            metadata={"rsi_value": last_row["rsi"]},
                        )
                    )

                elif last_row["rsi"] < self.params["oversold"]:
                    # Track signal
                    self.signal_counts["long"] += 1
                    self.signal_counts["total"] += 1
                    self.last_signal_time = pd.Timestamp.now()

                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"RSI LONG signal for {symbol}: RSI={last_row['rsi']:.2f}, total signals: {self.signal_counts['total']}")

                    signals.append(
                        self.create_signal(
                            symbol=symbol,
                            signal_type=SignalType.ENTRY_LONG,
                            strength=SignalStrength.STRONG,
                            order_type="market",
                            amount=self._calculate_dynamic_position_size(symbol),
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
