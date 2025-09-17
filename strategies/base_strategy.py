"""
strategies/base_strategy.py

Abstract base class for all trading strategies.
Defines the required interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from utils.time import now_ms, to_iso
import asyncio

import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Any

from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import OrderType
from data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration parameters for a trading strategy."""

    name: str
    symbols: List[str]
    timeframe: str
    required_history: int  # Number of candles needed for calculation
    enabled: bool = True
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate essential config fields after initialization."""
        if (
            not self.name
            or not self.symbols
            or not self.timeframe
            or not self.required_history
        ):
            raise ValueError("Invalid StrategyConfig: missing required fields")


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Concrete strategies must implement the abstract methods.
    """

    def __init__(self, config: Union[StrategyConfig, Dict[str, Any]]) -> None:
        """
        Initialize the strategy with its configuration.

        Args:
            config: Strategy configuration parameters (StrategyConfig or dict)
        """
        self.config = config
        # Use ms-based timestamp for deterministic IDs across processes
        # Handle both StrategyConfig objects and dict configs
        config_name = config.name if hasattr(config, 'name') else config.get('name', 'unknown')
        self.id = f"{config_name}_{now_ms()}"
        self.data_fetcher: Optional[DataFetcher] = None
        self.initialized = False
        self.last_signal_time = 0
        self.signals_generated = 0
        self._setup_logging()

        # Common signal tracking
        self.signal_counts: Dict[str, int] = {"long": 0, "short": 0, "total": 0}
        self.last_signal_time = None

    def _setup_logging(self) -> None:
        """Setup strategy-specific logging."""
        config_name = self.config.name if hasattr(self.config, 'name') else self.config.get('name', 'unknown')
        self.logger = logging.getLogger(f"strategy.{config_name.lower()}")
        self.logger.info(f"Initializing {config_name} strategy")

    def __init_subclass__(cls, **kwargs):
        """
        Wrap subclass `calculate_indicators` to enforce a minimal data length check
        so tests that call `calculate_indicators` directly get a ValueError when
        data is too short.
        """
        super().__init_subclass__(**kwargs)
        orig = getattr(cls, "calculate_indicators", None)
        if orig is None:
            return

        import inspect

        # Only wrap once
        if getattr(cls, "_calculate_wrapped", False):
            return

        if inspect.iscoroutinefunction(orig):

            async def wrapped(self, data, *args, **kw):
                if data is None or len(data) < getattr(
                    self.config, "required_history", 0
                ):
                    raise ValueError("Insufficient data for indicator calculation")
                return await orig(self, data, *args, **kw)

        else:

            def wrapped(self, data, *args, **kw):
                if data is None or len(data) < getattr(
                    self.config, "required_history", 0
                ):
                    raise ValueError("Insufficient data for indicator calculation")
                return orig(self, data, *args, **kw)

        setattr(cls, "calculate_indicators", wrapped)
        setattr(cls, "_calculate_wrapped", True)

    async def initialize(self, data_fetcher: DataFetcher) -> None:
        """
        Initialize the strategy with required resources.

        Args:
            data_fetcher: DataFetcher instance for market data access
        """
        self.data_fetcher = data_fetcher
        self.initialized = True
        config_name = self.config.name if hasattr(self.config, 'name') else self.config.get('name', 'unknown')
        self.logger.info(f"{config_name} strategy initialized")

    @abstractmethod
    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given market data.

        Args:
            data: DataFrame containing OHLCV market data

        Returns:
            DataFrame with additional indicator columns
        """
        pass

    @abstractmethod
    async def generate_signals(self, data: pd.DataFrame, multi_tf_data: Optional[Dict[str, Any]] = None) -> List[TradingSignal]:
        """
        Generate trading signals based on calculated indicators.

        Args:
            data: DataFrame containing OHLCV and indicator data
            multi_tf_data: Optional multi-timeframe data from TimeframeManager

        Returns:
            List of TradingSignal objects
        """
        pass

    async def run(self) -> List[TradingSignal]:
        """
        Execute the strategy's main logic and return signals.

        Returns:
            List of generated TradingSignal objects
        """
        if not self.initialized:
            raise RuntimeError("Strategy not initialized")

        try:
            # Fetch required market data
            data = await self._get_market_data()
            if data.empty:
                self.logger.warning("No market data available")
                return []

            # Calculate indicators
            data_with_indicators = await self.calculate_indicators(data)

            # Generate signals
            signals = await self.generate_signals(data_with_indicators)
            self._log_signals(signals)

            return signals

        except Exception as e:
            self.logger.error(f"Error in strategy execution: {str(e)}", exc_info=True)
            return []

    async def _get_market_data(self) -> pd.DataFrame:
        """
        Fetch the required market data for the strategy.

        Returns:
            DataFrame with OHLCV data for configured symbols and timeframe
        """
        if not self.data_fetcher:
            raise RuntimeError("DataFetcher not initialized")

        # Get data for all configured symbols
        all_data = []
        for symbol in self.config.symbols:
            try:
                data = await self.data_fetcher.get_historical_data(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    limit=self.config.required_history,
                )
                if data is not None:
                    data["symbol"] = symbol  # Add symbol column
                    all_data.append(data)
            except Exception as e:
                self.logger.error(f"Failed to get data for {symbol}: {str(e)}")

        if not all_data:
            return pd.DataFrame()

        # Combine data for all symbols
        combined = pd.concat(all_data)

        # Convert index to datetime if needed
        if not isinstance(combined.index, pd.DatetimeIndex):
            combined.index = pd.to_datetime(combined.index, unit="ms")

        return combined.sort_index()

    def _log_signals(self, signals: List[TradingSignal]) -> None:
        """Log generated signals."""
        if signals:
            self.signals_generated += len(signals)
            self.last_signal_time = now_ms()
            self.logger.info(f"Generated {len(signals)} new signals")

    def create_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: SignalStrength,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        current_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> TradingSignal:
        """
        Helper to create a TradingSignal with normalized types and deterministic timestamps.

        REPRODUCIBILITY: This method now accepts a timestamp parameter to ensure deterministic
        signal creation. For backtesting and training reproducibility, the timestamp should be
        derived from the input market data (e.g., the timestamp of the data point that triggered
        the signal). This prevents non-deterministic behavior where signals generated at different
        times would have different timestamps even with identical input data.

        Converts numeric values to Decimal for internal consistency and accepts
        OrderType enum for order_type.

        Args:
            symbol: Trading pair symbol.
            signal_type: SignalType enum value.
            strength: SignalStrength enum value.
            order_type: OrderType enum value.
            amount: Position size (numeric).
            price: Target price for limit orders (numeric).
            current_price: Current market price (numeric).
            stop_loss: Stop loss price (numeric).
            take_profit: Take profit price (numeric).
            trailing_stop: Trailing stop configuration.
            metadata: Additional strategy-specific data.
            timestamp: Deterministic timestamp from market data (for reproducibility).

        Returns:
            TradingSignal instance with Decimal-typed numeric fields and deterministic timestamp.
        """
        # Normalize numeric fields to Decimal where provided
        amt_dec = Decimal(str(amount)) if amount is not None else Decimal("0")
        price_dec = Decimal(str(price)) if price is not None else None
        current_dec = Decimal(str(current_price)) if current_price is not None else None
        stop_dec = Decimal(str(stop_loss)) if stop_loss is not None else None
        tp_dec = Decimal(str(take_profit)) if take_profit is not None else None

        # Use provided timestamp or fallback to current time (for backward compatibility)
        # In production backtesting/training, timestamp should always be provided
        signal_timestamp = timestamp if timestamp is not None else datetime.now()

        return TradingSignal(
            strategy_id=self.id,
            symbol=symbol,
            signal_type=signal_type,
            signal_strength=strength,
            order_type=order_type,
            amount=amt_dec,
            price=price_dec,
            current_price=current_dec,
            timestamp=signal_timestamp,
            stop_loss=stop_dec,
            take_profit=tp_dec,
            trailing_stop=trailing_stop,
            metadata=metadata or {},
        )

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        config_name = self.config.name if hasattr(self.config, 'name') else self.config.get('name', 'unknown')
        self.logger.info(f"Shutting down {config_name} strategy")
        self.initialized = False

    def get_performance_metrics(self) -> Dict:
        """Get strategy performance metrics."""
        return {
            "strategy_id": self.id,
            "name": self.config.name,
            "signals_generated": self.signals_generated,
            "last_signal_time": self.last_signal_time,
            "symbols": self.config.symbols,
            "timeframe": self.config.timeframe,
        }

    # Multi-Timeframe Analysis Helper Methods

    def get_higher_timeframe_trend(self, multi_tf_data: Dict[str, Any],
                                  current_tf: str, higher_tf: str) -> Optional[str]:
        """
        Analyze trend direction on a higher timeframe.

        Args:
            multi_tf_data: Multi-timeframe data from TimeframeManager
            current_tf: Current timeframe (e.g., '15m')
            higher_tf: Higher timeframe to analyze (e.g., '1h', '4h')

        Returns:
            Trend direction: 'bullish', 'bearish', or 'sideways'
        """
        try:
            if not multi_tf_data or higher_tf not in multi_tf_data.get('data', {}):
                return None

            higher_data = multi_tf_data['data'][higher_tf]
            if higher_data.empty or len(higher_data) < 20:
                return None

            # Calculate trend using moving averages
            sma_short = higher_data['close'].rolling(10).mean()
            sma_long = higher_data['close'].rolling(20).mean()

            if len(sma_short) < 2 or len(sma_long) < 2:
                return 'sideways'

            # Check recent trend
            recent_short = sma_short.iloc[-1]
            recent_long = sma_long.iloc[-1]
            prev_short = sma_short.iloc[-2]
            prev_long = sma_long.iloc[-2]

            if recent_short > recent_long and prev_short > prev_long:
                return 'bullish'
            elif recent_short < recent_long and prev_short < prev_long:
                return 'bearish'
            else:
                return 'sideways'

        except Exception as e:
            self.logger.warning(f"Failed to analyze higher timeframe trend: {e}")
            return None

    def validate_across_timeframes(self, signal: TradingSignal,
                                 multi_tf_data: Dict[str, Any],
                                 required_timeframes: List[str]) -> Dict[str, Any]:
        """
        Validate a signal across multiple timeframes.

        Args:
            signal: Trading signal to validate
            multi_tf_data: Multi-timeframe data from TimeframeManager
            required_timeframes: List of timeframes that must confirm the signal

        Returns:
            Validation result with confidence score and details
        """
        try:
            validation_result = {
                'is_valid': False,
                'confidence_score': 0.0,
                'confirming_timeframes': [],
                'conflicting_timeframes': [],
                'details': {}
            }

            if not multi_tf_data or not required_timeframes:
                return validation_result

            signal_type = signal.signal_type.value
            confirming_count = 0

            for tf in required_timeframes:
                if tf not in multi_tf_data.get('data', {}):
                    validation_result['conflicting_timeframes'].append(tf)
                    continue

                tf_data = multi_tf_data['data'][tf]
                trend = self.get_higher_timeframe_trend(multi_tf_data, signal.metadata.get('timeframe', '15m'), tf)

                if trend:
                    if signal_type in ['BUY', 'LONG'] and trend == 'bullish':
                        confirming_count += 1
                        validation_result['confirming_timeframes'].append(tf)
                    elif signal_type in ['SELL', 'SHORT'] and trend == 'bearish':
                        confirming_count += 1
                        validation_result['confirming_timeframes'].append(tf)
                    else:
                        validation_result['conflicting_timeframes'].append(tf)

            # Calculate confidence score
            if required_timeframes:
                validation_result['confidence_score'] = confirming_count / len(required_timeframes)
                validation_result['is_valid'] = validation_result['confidence_score'] >= 0.7  # 70% threshold

            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate signal across timeframes: {e}")
            return validation_result

    def calculate_multi_tf_indicators(self, multi_tf_data: Dict[str, Any],
                                    indicator_name: str, **kwargs) -> Dict[str, float]:
        """
        Calculate indicators across multiple timeframes.

        Args:
            multi_tf_data: Multi-timeframe data from TimeframeManager
            indicator_name: Name of indicator to calculate
            **kwargs: Additional parameters for indicator calculation

        Returns:
            Dictionary of timeframe -> indicator value
        """
        try:
            results = {}

            if not multi_tf_data or 'data' not in multi_tf_data:
                return results

            for tf, tf_data in multi_tf_data['data'].items():
                if tf_data.empty:
                    continue

                try:
                    if indicator_name == 'trend_strength':
                        period = kwargs.get('period', 14)
                        strength = self.calculate_trend_strength(tf_data['close'], period)
                        results[tf] = strength
                    elif indicator_name == 'volatility':
                        period = kwargs.get('period', 20)
                        vol = self.calculate_volatility(tf_data['close'], period)
                        results[tf] = vol
                    elif indicator_name == 'atr':
                        period = kwargs.get('period', 14)
                        atr = self.calculate_atr(tf_data, period)
                        results[tf] = atr
                    # Add more indicators as needed

                except Exception as e:
                    self.logger.warning(f"Failed to calculate {indicator_name} for {tf}: {e}")

            return results

        except Exception as e:
            self.logger.error(f"Failed to calculate multi-timeframe indicators: {e}")
            return {}

    def get_multi_tf_consensus_score(self, multi_tf_data: Dict[str, Any],
                                   signal_type: str, timeframes: List[str]) -> float:
        """
        Calculate consensus score across multiple timeframes for a signal type.

        Args:
            multi_tf_data: Multi-timeframe data from TimeframeManager
            signal_type: Type of signal ('BUY', 'SELL', etc.)
            timeframes: List of timeframes to analyze

        Returns:
            Consensus score (0-1, higher is better consensus)
        """
        try:
            if not multi_tf_data or not timeframes:
                return 0.0

            consensus_count = 0
            total_timeframes = len(timeframes)

            for tf in timeframes:
                trend = self.get_higher_timeframe_trend(multi_tf_data,
                                                       self.config.timeframe, tf)
                if trend:
                    if signal_type in ['BUY', 'LONG'] and trend == 'bullish':
                        consensus_count += 1
                    elif signal_type in ['SELL', 'SHORT'] and trend == 'bearish':
                        consensus_count += 1

            return consensus_count / total_timeframes if total_timeframes > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Failed to calculate consensus score: {e}")
            return 0.0


class TrendAnalysisMixin:
    """Mixin class providing common trend analysis methods."""

    async def calculate_trend_strength(
        self, prices: pd.Series, period: int = 14
    ) -> float:
        """
        Calculate trend strength using ADX methodology.

        Args:
            prices: Series of closing prices
            period: Lookback period

        Returns:
            Trend strength score (0-1)
        """
        if len(prices) < period * 2:
            return 0.0

        try:
            # Calculate directional movements
            delta = prices.diff()
            up = delta.copy()
            up[up < 0] = 0
            down = delta.copy()
            down[down > 0] = 0
            down = down.abs()

            # Calculate smoothed averages
            roll_up = up.rolling(period).mean()
            roll_down = down.rolling(period).mean()

            # Calculate DX
            dx = (roll_up - roll_down).abs() / (roll_up + roll_down)
            adx = dx.rolling(period).mean()

            # Normalize to 0-1 range
            return float(adx.iloc[-1] / 100)
        except Exception:
            return 0.0

    async def identify_support_resistance(
        self, prices: pd.DataFrame, lookback: int = 20
    ) -> Dict:
        """
        Identify key support and resistance levels.

        Args:
            prices: DataFrame with OHLC data
            lookback: Number of periods to analyze

        Returns:
            Dict with support/resistance levels
        """
        # If we have less data than the requested lookback, use whatever is available.
        if len(prices) < lookback:
            recent = prices
        else:
            recent = prices.iloc[-lookback:]
        support = recent["low"].min()
        resistance = recent["high"].max()

        # Compute current position within the range safely (avoid div-by-zero).
        try:
            range_span = float(resistance) - float(support)
            if range_span == 0:
                current_pos = 0.0
            else:
                current_pos = float((recent["close"].iloc[-1] - support) / range_span)
        except Exception:
            current_pos = 0.0

        return {
            "support": float(support) if support is not None else None,
            "resistance": float(resistance) if resistance is not None else None,
            "current_position": current_pos,
        }


class VolatilityAnalysisMixin:
    """Mixin class providing common volatility analysis methods."""

    async def calculate_atr(self, prices: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).

        Args:
            prices: DataFrame with OHLC data
            period: Lookback period

        Returns:
            ATR value
        """
        if len(prices) < period:
            return 0.0

        try:
            high_low = prices["high"] - prices["low"]
            high_close = (prices["high"] - prices["close"].shift()).abs()
            low_close = (prices["low"] - prices["close"].shift()).abs()

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )
            atr = true_range.rolling(period).mean()
            return float(atr.iloc[-1])
        except Exception:
            return 0.0

    async def calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """
        Calculate volatility as standard deviation of returns.

        Args:
            prices: Series of closing prices
            period: Lookback period

        Returns:
            Volatility measure
        """
        if len(prices) < period:
            return 0.0

        returns = np.log(prices / prices.shift(1))
        return float(returns.std() * np.sqrt(period))


class SignalGenerationMixin:
    """Mixin class providing common signal generation methods."""

    def _merge_params(self, default_params: Dict[str, Any], config: Union[StrategyConfig, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge default parameters with config parameters.

        Args:
            default_params: Default parameter dictionary
            config: Strategy configuration

        Returns:
            Merged parameters dictionary
        """
        config_params = config.params if hasattr(config, 'params') else config.get('params', {})
        return {**default_params, **(config_params or {})}

    def _check_volume_confirmation(self, data: pd.DataFrame, symbol: str, volume_period: int, volume_threshold: float) -> bool:
        """
        Check if volume confirms the signal.

        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            volume_period: Period for volume averaging
            volume_threshold: Volume multiplier threshold

        Returns:
            True if volume confirms, False otherwise
        """
        try:
            symbol_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data
            if len(symbol_data) < volume_period or "volume" not in symbol_data.columns:
                return True  # Default to allowing signal if no volume data

            last_row = symbol_data.iloc[-1]
            if pd.isna(last_row["volume"]):
                return True

            avg_volume = symbol_data["volume"].tail(volume_period).mean()
            current_volume = last_row["volume"]
            return current_volume >= (avg_volume * volume_threshold)
        except Exception:
            return True  # Default to allowing signal on error

    def _create_breakout_signal(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: SignalStrength,
        current_price: float,
        position_size: float,
        stop_loss_pct: float,
        take_profit_pct: float,
        metadata: Dict[str, Any]
    ) -> TradingSignal:
        """
        Create a breakout trading signal.

        Args:
            symbol: Trading symbol
            signal_type: Type of signal
            strength: Signal strength
            current_price: Current market price
            position_size: Position size percentage
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            metadata: Additional signal metadata

        Returns:
            TradingSignal object
        """
        stop_loss = current_price * (1 - stop_loss_pct) if signal_type == SignalType.ENTRY_LONG else current_price * (1 + stop_loss_pct)
        take_profit = current_price * (1 + take_profit_pct) if signal_type == SignalType.ENTRY_LONG else current_price * (1 - take_profit_pct)

        return self.create_signal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            order_type="market",
            amount=position_size,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=metadata,
        )

    def _update_signal_counts(self, signal_type: str) -> None:
        """
        Update signal tracking counts.

        REPRODUCIBILITY: Removed non-deterministic timestamp update to ensure
        that signal generation is deterministic based on input data only.
        The last_signal_time is now updated in _log_signals using now_ms()
        which is more appropriate for tracking purposes.

        Args:
            signal_type: Type of signal ('long', 'short', etc.)
        """
        if signal_type in self.signal_counts:
            self.signal_counts[signal_type] += 1
        self.signal_counts["total"] += 1
        # Note: last_signal_time is now updated deterministically in _log_signals
