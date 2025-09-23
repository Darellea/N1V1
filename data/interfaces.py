"""
Abstract interfaces for the data module.

This module defines abstract base classes that enable dependency injection
and decoupling of components in the data module.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class IDataFetcher(ABC):
    """
    Abstract interface for data fetching operations.

    This interface defines the contract that all data fetchers must implement,
    enabling dependency injection and easier testing with mocks.
    """

    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 1000,
        since: Optional[int] = None,
        force_fresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            since: Timestamp in ms of earliest candle to fetch
            force_fresh: Bypass cache and fetch fresh data

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    async def get_realtime_data(
        self,
        symbols: List[str],
        tickers: bool = True,
        orderbooks: bool = False,
        depth: int = 5
    ) -> Dict:
        """
        Get real-time market data for multiple symbols.

        Args:
            symbols: List of trading pair symbols
            tickers: Whether to fetch ticker data
            orderbooks: Whether to fetch order book data
            depth: Order book depth if fetching

        Returns:
            Dictionary of real-time data for each symbol
        """
        pass

    @abstractmethod
    async def get_multiple_historical_data(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 1000,
        since: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols concurrently.

        Args:
            symbols: List of trading pair symbols
            timeframe: Timeframe string
            limit: Number of candles per symbol
            since: Timestamp in ms of earliest candle

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup resources."""
        pass
