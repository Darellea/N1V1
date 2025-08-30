"""
strategies/base_strategy.py

Abstract base class for all trading strategies.
Defines the required interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
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

    def __post_init__(self):
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

    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy with its configuration.

        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        # Use ms-based timestamp for deterministic IDs across processes
        self.id = f"{config.name}_{now_ms()}"
        self.data_fetcher: Optional[DataFetcher] = None
        self.initialized = False
        self.last_signal_time = 0
        self.signals_generated = 0
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup strategy-specific logging."""
        self.logger = logging.getLogger(f"strategy.{self.config.name.lower()}")
        self.logger.info(f"Initializing {self.config.name} strategy")

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
        self.logger.info(f"{self.config.name} strategy initialized")

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
    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on calculated indicators.

        Args:
            data: DataFrame containing OHLCV and indicator data

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
    ) -> TradingSignal:
        """
        Helper to create a TradingSignal with normalized types.

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

        Returns:
            TradingSignal instance with Decimal-typed numeric fields.
        """
        # Normalize numeric fields to Decimal where provided
        amt_dec = Decimal(str(amount)) if amount is not None else Decimal("0")
        price_dec = Decimal(str(price)) if price is not None else None
        current_dec = Decimal(str(current_price)) if current_price is not None else None
        stop_dec = Decimal(str(stop_loss)) if stop_loss is not None else None
        tp_dec = Decimal(str(take_profit)) if take_profit is not None else None

        return TradingSignal(
            strategy_id=self.id,
            symbol=symbol,
            signal_type=signal_type,
            signal_strength=strength,
            order_type=order_type,
            amount=amt_dec,
            price=price_dec,
            current_price=current_dec,
            stop_loss=stop_dec,
            take_profit=tp_dec,
            trailing_stop=trailing_stop,
            metadata=metadata or {},
        )

    async def shutdown(self) -> None:
        """Cleanup strategy resources."""
        self.logger.info(f"Shutting down {self.config.name} strategy")
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
