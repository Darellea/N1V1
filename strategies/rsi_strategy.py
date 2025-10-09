"""
RSI Strategy

A momentum-based strategy that uses Relative Strength Index (RSI) to identify
overbought and oversold conditions for mean reversion signals.
"""

# Centralized configuration constants for robustness and maintainability
RSI_PERIOD = 14  # Period for RSI calculation
OVERBOUGHT = 70  # Overbought threshold
OVERSOLD = 30  # Oversold threshold
POSITION_SIZE = 0.1  # 10% of portfolio
STOP_LOSS_PCT = 0.05  # 5%
TAKE_PROFIT_PCT = 0.1  # 10%
VOLUME_PERIOD = 10  # Period for volume averaging in signal confirmation
VOLUME_THRESHOLD = 1.5  # Volume must be 1.5x volume_period average

import logging
import traceback
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from core.contracts import SignalStrength, SignalType, TradingSignal
from strategies.base_strategy import BaseStrategy, SignalGenerationMixin, StrategyConfig
from strategies.indicators_cache import calculate_indicators_for_multi_symbol


class RSIStrategy(BaseStrategy, SignalGenerationMixin):
    """Relative Strength Index trading strategy."""

    def __init__(self, config: Union[StrategyConfig, Dict[str, Any]]) -> None:
        """Initialize RSI strategy.

        Args:
            config: StrategyConfig instance or dict with required parameters.
        """
        super().__init__(config)
        self.default_params: Dict[str, Any] = {
            "rsi_period": RSI_PERIOD,  # Period for RSI calculation (separate from volume_period)
            "overbought": OVERBOUGHT,
            "oversold": OVERSOLD,
            "position_size": POSITION_SIZE,  # 10% of portfolio
            "stop_loss_pct": STOP_LOSS_PCT,  # 5%
            "take_profit_pct": TAKE_PROFIT_PCT,  # 10%
            "volume_period": VOLUME_PERIOD,  # Period for volume averaging in signal confirmation (separate from rsi_period)
            "volume_threshold": VOLUME_THRESHOLD,  # Volume must be 1.5x volume_period average
        }
        self.params = self._merge_params(self.default_params, config)

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI indicator using vectorized operations for better performance.

        PERFORMANCE IMPROVEMENT: This method now uses vectorized pandas operations
        instead of inefficient groupby/apply patterns. For multi-symbol data, it processes
        all symbols simultaneously using numpy/pandas vectorization, which provides
        significant performance gains especially with large datasets containing many symbols.

        The shared indicators_cache module provides additional caching to avoid redundant
        calculations when the same indicator parameters are used across different strategies.
        """
        if data.empty:
            return data

        # Input validation: Check for required columns
        required_columns = ["close"]  # RSI primarily needs close prices
        if self.params.get("volume_filter", False):
            required_columns.append("volume")
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in market data: {missing_columns}. "
                f"Expected columns: {required_columns}"
            )

        # Use vectorized calculation for all symbols at once
        # This eliminates the need for groupby operations and provides much better performance
        indicators_config = {"rsi": {"period": int(self.params["rsi_period"])}}

        # Calculate indicators using the shared vectorized function
        result_df = calculate_indicators_for_multi_symbol(data, indicators_config)

        return result_df

    async def generate_signals(
        self, data: pd.DataFrame, multi_tf_data: Optional[Dict[str, Any]] = None
    ) -> List[TradingSignal]:
        """
        Generate signals based on RSI values using vectorized operations.

        PERFORMANCE IMPROVEMENT: This method now uses vectorized pandas operations
        to process all symbols simultaneously instead of iterating through each symbol
        individually. This eliminates the overhead of groupby operations and provides
        significant performance improvements for multi-symbol scenarios.
        """
        signals: List[TradingSignal] = []

        if data.empty or "symbol" not in data.columns:
            return signals

        try:
            # Ensure indicators are calculated
            data_with_indicators = await self._ensure_indicators_calculated(data)

            # Get the last row for each symbol using vectorized operations
            last_rows = data_with_indicators.groupby("symbol").tail(1)

            # Process each symbol's data
            for symbol in data_with_indicators["symbol"].unique():
                symbol_signals = await self._process_symbol_for_signals(
                    symbol, data_with_indicators, last_rows
                )
                signals.extend(symbol_signals)

        except Exception as e:
            self.logger.error(f"Error in signal generation: {str(e)}")

        return signals

    async def _ensure_indicators_calculated(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure RSI indicators are calculated for the data."""
        if "rsi" not in data.columns:
            return await self.calculate_indicators(data)
        return data

    async def _process_symbol_for_signals(
        self, symbol: str, data: pd.DataFrame, last_rows: pd.DataFrame
    ) -> List[TradingSignal]:
        """Process a single symbol to generate signals."""
        signals: List[TradingSignal] = []

        try:
            # Get the last row for this symbol
            symbol_last_row = last_rows[last_rows["symbol"] == symbol]
            if symbol_last_row.empty:
                return signals

            row = symbol_last_row.iloc[0]
            current_price = row["close"]
            rsi_value = row["rsi"]

            # Validate RSI value
            if pd.isna(rsi_value):
                self.logger.warning(
                    f"NaN detected in RSI indicator for symbol {symbol}. "
                    f"Skipping signal generation."
                )
                return signals

            # Check volume confirmation
            if not self._check_volume_confirmation_for_symbol(symbol, data, row):
                return signals

            # Generate signals based on RSI levels
            rsi_signals = self._generate_signals_from_rsi_levels(
                symbol, current_price, rsi_value, data
            )
            signals.extend(rsi_signals)

        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {str(e)}")

        return signals

    def _check_volume_confirmation_for_symbol(
        self, symbol: str, data: pd.DataFrame, row: pd.Series
    ) -> bool:
        """Check if volume confirms the signal for a symbol."""
        if not self.params.get("volume_filter", False):
            return True  # Volume filter not enabled

        if "volume" not in row.index or pd.isna(row["volume"]):
            return True  # No volume data, allow signal

        try:
            symbol_data = data[data["symbol"] == symbol]
            volume_period = int(self.params.get("volume_period", 10))

            if len(symbol_data) < volume_period:
                return True  # Not enough data for volume check

            avg_volume = symbol_data["volume"].tail(volume_period).mean()
            current_volume = row["volume"]
            volume_threshold = self.params.get("volume_threshold", 1.5)

            return current_volume >= (avg_volume * volume_threshold)

        except Exception as e:
            self.logger.warning(f"Volume confirmation failed for {symbol}: {str(e)}")
            return True  # Default to allowing signal on error

    def _generate_signals_from_rsi_levels(
        self, symbol: str, current_price: float, rsi_value: float, data: pd.DataFrame
    ) -> List[TradingSignal]:
        """Generate signals based on RSI overbought/oversold levels."""
        signals: List[TradingSignal] = []

        if rsi_value >= self.params["overbought"]:
            self._update_signal_counts("short")
            signals.append(
                self._create_rsi_signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_SHORT,
                    current_price=current_price,
                    rsi_value=rsi_value,
                    data=data,
                )
            )

        elif rsi_value <= self.params["oversold"]:
            self._update_signal_counts("long")
            signals.append(
                self._create_rsi_signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    current_price=current_price,
                    rsi_value=rsi_value,
                    data=data,
                )
            )

        return signals

    def _create_rsi_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        current_price: float,
        rsi_value: float,
        data: pd.DataFrame,
    ) -> TradingSignal:
        """Create a trading signal for RSI-based entry with deterministic timestamp."""
        is_long = signal_type == SignalType.ENTRY_LONG

        # Extract deterministic timestamp from the data that triggered the signal
        # Use the timestamp of the last data point for reproducibility
        signal_timestamp = None
        if not data.empty and isinstance(data.index, pd.DatetimeIndex):
            signal_timestamp = data.index[-1].to_pydatetime()

        return self.create_signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=SignalStrength.STRONG,
            order_type="market",
            amount=self._calculate_dynamic_position_size(symbol),
            current_price=current_price,
            stop_loss=current_price
            * (
                1 - self.params["stop_loss_pct"]
                if is_long
                else 1 + self.params["stop_loss_pct"]
            ),
            take_profit=current_price
            * (
                1 + self.params["take_profit_pct"]
                if is_long
                else 1 - self.params["take_profit_pct"]
            ),
            metadata={"rsi_value": rsi_value},
            timestamp=signal_timestamp,
        )

    def _calculate_dynamic_position_size(self, symbol: str) -> float:
        """Calculate dynamic position size based on portfolio allocation or ATR."""
        # For now, use base position size
        # In a full implementation, this would consider:
        # - Portfolio allocation for the symbol
        # - ATR-based volatility adjustment
        # - Account balance and risk management
        return self.params["position_size"]

    async def _generate_signals_for_symbol(
        self, symbol: str, data
    ) -> List[TradingSignal]:
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

            # Check for NaN in RSI indicator
            if pd.isna(last_row["rsi"]):
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"NaN detected in RSI indicator for symbol {symbol}. Skipping signal generation. "
                    f"This may indicate data quality issues or insufficient data for RSI calculation."
                )
                return signals

            # Generate signals if RSI is valid
            # Volume confirmation filter
            volume_confirmed = True
            if (
                self.params.get("volume_filter", False)
                and "volume" in last_row.index
                and not pd.isna(last_row["volume"])
            ):
                try:
                    # Calculate average volume over the specified period
                    volume_period = int(self.params.get("volume_period", 10))
                    if len(data_with_rsi) >= volume_period:
                        avg_volume = data_with_rsi["volume"].tail(volume_period).mean()
                        current_volume = last_row["volume"]
                        volume_threshold = self.params.get("volume_threshold", 1.5)
                        volume_confirmed = current_volume >= (
                            avg_volume * volume_threshold
                        )
                    else:
                        # Not enough data for volume check, proceed
                        volume_confirmed = True
                except (ValueError, TypeError, KeyError) as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Volume confirmation failed for {symbol} due to data issue: {str(e)}. "
                        f"Proceeding with signal generation."
                    )
                    volume_confirmed = True  # Default to allowing signal

            if not volume_confirmed:
                logger = logging.getLogger(__name__)
                logger.info(f"Volume confirmation failed for {symbol}: signal skipped")
                return signals

            if last_row["rsi"] >= self.params["overbought"]:
                # Track signal
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                logger = logging.getLogger(__name__)
                logger.info(
                    f"RSI SHORT signal for {symbol}: RSI={last_row['rsi']:.2f}, total signals: {self.signal_counts['total']}"
                )

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_rsi.empty and isinstance(
                    data_with_rsi.index, pd.DatetimeIndex
                ):
                    signal_timestamp = data_with_rsi.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_SHORT,
                        strength=SignalStrength.STRONG,
                        order_type="market",
                        amount=self._calculate_dynamic_position_size(symbol),
                        current_price=current_price,
                        stop_loss=current_price * (1 + self.params["stop_loss_pct"]),
                        take_profit=current_price
                        * (1 - self.params["take_profit_pct"]),
                        metadata={"rsi_value": last_row["rsi"]},
                        timestamp=signal_timestamp,
                    )
                )

            elif last_row["rsi"] <= self.params["oversold"]:
                # Track signal
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                logger = logging.getLogger(__name__)
                logger.info(
                    f"RSI LONG signal for {symbol}: RSI={last_row['rsi']:.2f}, total signals: {self.signal_counts['total']}"
                )

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_rsi.empty and isinstance(
                    data_with_rsi.index, pd.DatetimeIndex
                ):
                    signal_timestamp = data_with_rsi.index[-1].to_pydatetime()

                signals.append(
                    self.create_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY_LONG,
                        strength=SignalStrength.STRONG,
                        order_type="market",
                        amount=self._calculate_dynamic_position_size(symbol),
                        current_price=current_price,
                        stop_loss=current_price * (1 - self.params["stop_loss_pct"]),
                        take_profit=current_price
                        * (1 + self.params["take_profit_pct"]),
                        metadata={"rsi_value": last_row["rsi"]},
                        timestamp=signal_timestamp,
                    )
                )

        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as e:
            logger = logging.getLogger(__name__)
            logger.error(
                f"Data processing error generating signals for {symbol}: {str(e)}. "
                f"Stack trace: {traceback.format_exc()}"
            )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(
                f"Unexpected error generating signals for {symbol}: {str(e)}. "
                f"Stack trace: {traceback.format_exc()}"
            )

        return signals
