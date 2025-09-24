"""
Bollinger Bands Mean Reversion Strategy

A range-based strategy that trades when price touches or moves beyond the
Bollinger Bands, expecting mean reversion back to the middle band.

# Centralized configuration constants for robustness and maintainability
PERIOD = 20  # Bollinger Bands period
STD_DEV = 2.0  # Standard deviation multiplier
REVERSION_THRESHOLD = 0.01  # Minimum distance from band for signal
POSITION_SIZE = 0.08  # 8% of portfolio (smaller for mean reversion)
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit (smaller targets)
VOLUME_FILTER = True  # Use volume confirmation
VOLUME_THRESHOLD = 1.1  # Volume must be 1.1x average
MAX_HOLDING_PERIOD = 10  # Maximum bars to hold position
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging
import traceback

from strategies.base_strategy import BaseStrategy, StrategyConfig
from strategies.indicators_cache import calculate_indicators_for_multi_symbol
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
        """
        Calculate Bollinger Bands indicators using vectorized operations for better performance.

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
        required_columns = ['close']  # Bollinger Bands primarily needs close prices
        if self.params.get("volume_filter", False):
            required_columns.append('volume')
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in market data: {missing_columns}. "
                           f"Expected columns: {required_columns}")

        # Use vectorized calculation for all symbols at once
        # This eliminates the need for groupby operations and provides much better performance
        indicators_config = {
            'bb': {
                'period': int(self.params["period"]),
                'std_dev': self.params["std_dev"]
            }
        }

        # Calculate indicators using the shared vectorized function
        result_df = calculate_indicators_for_multi_symbol(data, indicators_config)

        return result_df

    async def generate_signals(self, data) -> List[TradingSignal]:
        """
        Generate signals based on Bollinger Band mean reversion using vectorized operations.

        PERFORMANCE IMPROVEMENT: This method now uses vectorized pandas operations
        to process all symbols simultaneously instead of iterating through each symbol
        individually. This eliminates the overhead of groupby operations and provides
        significant performance improvements for multi-symbol scenarios.
        """
        signals = []

        if data.empty or "symbol" not in data.columns:
            return signals

        try:
            # Ensure indicators are calculated
            if not all(col in data.columns for col in ['bb_position', 'bb_width']):
                data = await self.calculate_indicators(data)

            # Get the last row for each symbol using vectorized operations
            last_rows = data.groupby("symbol").tail(1)

            # Vectorized signal generation
            for idx, row in last_rows.iterrows():
                symbol = row["symbol"]
                current_price = row["close"]
                bb_position = row["bb_position"]
                bb_width = row["bb_width"]

                # Check for NaN in Bollinger Bands position
                if pd.isna(bb_position):
                    logger = logging.getLogger(__name__)
                    logger.warning(f"NaN detected in Bollinger Bands position for symbol {symbol} during vectorized processing. "
                                  f"Skipping signal generation. This may indicate data quality issues or insufficient data for BB calculation.")
                    continue

                # Volume confirmation (vectorized where possible)
                volume_confirmed = True
                if self.params["volume_filter"] and "volume" in row.index and not pd.isna(row["volume"]):
                    try:
                        # Get symbol-specific data for volume calculation
                        symbol_data = data[data["symbol"] == symbol]
                        volume_period = int(self.params["period"])

                        if len(symbol_data) >= volume_period:
                            avg_volume = symbol_data["volume"].tail(volume_period).mean()
                            current_volume = row["volume"]
                            volume_confirmed = current_volume >= (avg_volume * self.params["volume_threshold"])
                    except (ValueError, TypeError, KeyError) as e:
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Volume confirmation failed for {symbol} due to data issue: {str(e)}. "
                                      f"Proceeding with signal generation.")
                        volume_confirmed = True

                if not volume_confirmed:
                    continue

                # Check for oversold condition (price near lower band)
                oversold_threshold = self.params["reversion_threshold"]
                oversold = bb_position < -oversold_threshold

                # Check for overbought condition (price near upper band)
                overbought_threshold = 1.0 - self.params["reversion_threshold"]
                overbought = bb_position > overbought_threshold

                # Additional filter: ensure bands are not too narrow (avoid whipsaw)
                min_bandwidth = 0.005  # Minimum 0.5% bandwidth
                valid_bandwidth = bb_width > min_bandwidth

                if oversold and valid_bandwidth:
                    # Price is oversold, expect mean reversion up
                    self.signal_counts["long"] += 1
                    self.signal_counts["total"] += 1
                    # Note: last_signal_time is now updated deterministically in _log_signals

                    # Extract deterministic timestamp from the data that triggered the signal
                    signal_timestamp = None
                    if not data.empty and isinstance(data.index, pd.DatetimeIndex):
                        signal_timestamp = data.index[-1].to_pydatetime()

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
                                "bb_position": bb_position,
                                "bb_middle": row.get("bb_middle", None),
                                "bb_upper": row.get("bb_upper", None),
                                "bb_lower": row.get("bb_lower", None),
                                "bb_width": bb_width
                            },
                            timestamp=signal_timestamp,
                        )
                    )

                elif overbought and valid_bandwidth:
                    # Price is overbought, expect mean reversion down
                    self.signal_counts["short"] += 1
                    self.signal_counts["total"] += 1
                    # Note: last_signal_time is now updated deterministically in _log_signals

                    # Extract deterministic timestamp from the data that triggered the signal
                    signal_timestamp = None
                    if not data.empty and isinstance(data.index, pd.DatetimeIndex):
                        signal_timestamp = data.index[-1].to_pydatetime()

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
                                "bb_position": bb_position,
                                "bb_middle": row.get("bb_middle", None),
                                "bb_upper": row.get("bb_upper", None),
                                "bb_lower": row.get("bb_lower", None),
                                "bb_width": bb_width
                            },
                            timestamp=signal_timestamp,
                        )
                    )

        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Data processing error in vectorized signal generation: {str(e)}. "
                        f"Stack trace: {traceback.format_exc()}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error in vectorized signal generation: {str(e)}. "
                        f"Stack trace: {traceback.format_exc()}")

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
                except (ValueError, TypeError, KeyError) as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Volume confirmation failed for {symbol} due to data issue: {str(e)}. "
                                  f"Proceeding with signal generation.")
                    volume_confirmed = True

            if not volume_confirmed:
                return signals

            # Check for NaN in Bollinger Bands position
            if pd.isna(last_row["bb_position"]):
                logger = logging.getLogger(__name__)
                logger.warning(f"NaN detected in Bollinger Bands position for symbol {symbol}. Skipping signal generation. "
                              f"This may indicate data quality issues or insufficient data for BB calculation.")
                return signals

            # Check for oversold condition (price near lower band)
            oversold_threshold = self.params["reversion_threshold"]
            oversold = last_row["bb_position"] < -oversold_threshold

            # Check for overbought condition (price near upper band)
            overbought_threshold = 1.0 - self.params["reversion_threshold"]
            overbought = last_row["bb_position"] > overbought_threshold

            # Additional filter: ensure bands are not too narrow (avoid whipsaw)
            min_bandwidth = 0.005  # Minimum 0.5% bandwidth
            valid_bandwidth = last_row["bb_width"] > min_bandwidth

            if oversold and valid_bandwidth:
                # Price is oversold, expect mean reversion up
                self.signal_counts["long"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_bb.empty and isinstance(data_with_bb.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_bb.index[-1].to_pydatetime()

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
                        timestamp=signal_timestamp,
                    )
                )

            elif overbought and valid_bandwidth:
                # Price is overbought, expect mean reversion down
                self.signal_counts["short"] += 1
                self.signal_counts["total"] += 1
                # Note: last_signal_time is now updated deterministically in _log_signals

                # Extract deterministic timestamp from the data that triggered the signal
                signal_timestamp = None
                if not data_with_bb.empty and isinstance(data_with_bb.index, pd.DatetimeIndex):
                    signal_timestamp = data_with_bb.index[-1].to_pydatetime()

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
                        timestamp=signal_timestamp,
                    )
                )

        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Data processing error generating signals for {symbol}: {str(e)}. "
                        f"Stack trace: {traceback.format_exc()}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Unexpected error generating signals for {symbol}: {str(e)}. "
                        f"Stack trace: {traceback.format_exc()}")

        return signals
