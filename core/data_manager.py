"""
DataManager - Market data fetching and caching component.

Handles fetching market data from various sources, caching for performance,
and managing multi-timeframe data operations.
"""

import time
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages market data fetching, caching, and multi-timeframe operations.

    Responsibilities:
    - Market data fetching from DataFetcher
    - Caching with TTL for performance
    - Multi-timeframe data coordination
    - Data validation and error handling
    """

    def __init__(self, data_fetcher, timeframe_manager=None, cache_ttl: float = 60.0):
        """Initialize the DataManager.

        Args:
            data_fetcher: DataFetcher instance for market data
            timeframe_manager: Optional TimeframeManager for multi-timeframe data
            cache_ttl: Cache time-to-live in seconds
        """
        self.data_fetcher = data_fetcher
        self.timeframe_manager = timeframe_manager
        self.cache_ttl = cache_ttl

        # Market data cache
        self.market_data_cache: Dict[str, Any] = {}
        self.cache_timestamp: float = 0.0

        # Trading pairs to monitor
        self.pairs: List[str] = []

        # Portfolio mode flag
        self.portfolio_mode: bool = False

    def set_trading_pairs(self, pairs: List[str], portfolio_mode: bool = False):
        """Set the trading pairs to monitor."""
        self.pairs = pairs
        self.portfolio_mode = portfolio_mode
        logger.info(f"DataManager configured for {len(pairs)} pairs in {'portfolio' if portfolio_mode else 'single'} mode")

    async def fetch_market_data(self) -> Dict[str, Any]:
        """Fetch market data with caching and multi-timeframe support."""
        current_time = time.time()

        # Return cached data if still valid
        if self.market_data_cache and (current_time - self.cache_timestamp) < self.cache_ttl:
            logger.debug("Using cached market data")
            return self.market_data_cache

        # Fetch fresh data
        market_data = await self._fetch_fresh_market_data()
        multi_timeframe_data = await self._fetch_multi_timeframe_data()

        # Combine single-timeframe and multi-timeframe data
        combined_data = self._combine_market_data(market_data, multi_timeframe_data)

        # Cache the fetched data
        self.market_data_cache = combined_data
        self.cache_timestamp = current_time

        logger.debug("Fetched and cached fresh market data")
        return self.market_data_cache

    async def _fetch_fresh_market_data(self) -> Dict[str, Any]:
        """Fetch fresh market data from the data fetcher."""
        market_data = {}

        try:
            if self.portfolio_mode and hasattr(self.data_fetcher, "get_realtime_data"):
                market_data = await self.data_fetcher.get_realtime_data(self.pairs)
            elif not self.portfolio_mode and hasattr(self.data_fetcher, "get_historical_data"):
                symbol = self.pairs[0] if self.pairs else None
                if symbol:
                    df = await self.data_fetcher.get_historical_data(
                        symbol=symbol,
                        timeframe=self._get_default_timeframe(),
                        limit=100,
                    )
                    market_data = {symbol: df}
            else:
                if hasattr(self.data_fetcher, "get_multiple_historical_data"):
                    market_data = await self.data_fetcher.get_multiple_historical_data(self.pairs)

        except Exception as e:
            logger.exception(f"Failed to fetch market data: {e}")
            # Return cached data if available as fallback
            if self.market_data_cache:
                logger.warning("Using stale cached data due to fetch failure")
                return self.market_data_cache

        return market_data

    async def _fetch_multi_timeframe_data(self) -> Dict[str, Any]:
        """Fetch multi-timeframe data if timeframe manager is available."""
        multi_timeframe_data = {}

        if not self.timeframe_manager or not self.pairs:
            return multi_timeframe_data

        for symbol in self.pairs:
            try:
                synced_data = await self.timeframe_manager.fetch_multi_timeframe_data(symbol)
                if synced_data:
                    multi_timeframe_data[symbol] = synced_data
                    logger.debug(f"Fetched multi-timeframe data for {symbol}")
                else:
                    logger.warning(f"Failed to fetch multi-timeframe data for {symbol}")
            except Exception as e:
                logger.warning(f"Error fetching multi-timeframe data for {symbol}: {e}")

        return multi_timeframe_data

    def _combine_market_data(self, market_data: Dict[str, Any],
                           multi_timeframe_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine single-timeframe and multi-timeframe data."""
        combined_data = market_data.copy()

        for symbol, synced_data in multi_timeframe_data.items():
            if symbol in combined_data:
                # Add multi-timeframe data to existing symbol data
                combined_data[symbol] = {
                    'single_timeframe': combined_data[symbol],
                    'multi_timeframe': synced_data
                }
            else:
                # Only multi-timeframe data available
                combined_data[symbol] = {
                    'multi_timeframe': synced_data
                }

        return combined_data

    def _get_default_timeframe(self) -> str:
        """Get default timeframe for single-pair mode."""
        # This could be configurable in the future
        return "1h"

    def clear_cache(self):
        """Clear the market data cache."""
        self.market_data_cache = {}
        self.cache_timestamp = 0.0
        logger.debug("Market data cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            "cache_size": len(self.market_data_cache),
            "cache_age": time.time() - self.cache_timestamp if self.cache_timestamp > 0 else 0,
            "cache_ttl": self.cache_ttl,
            "is_cache_valid": (time.time() - self.cache_timestamp) < self.cache_ttl if self.cache_timestamp > 0 else False
        }

    def extract_multi_timeframe_data(self, market_data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """
        Extract multi-timeframe data for a specific symbol from market data.

        Args:
            market_data: Combined market data dictionary
            symbol: Symbol to extract data for

        Returns:
            Multi-timeframe data or None if not available
        """
        try:
            if not market_data or symbol not in market_data:
                return None

            symbol_data = market_data[symbol]

            # Check if symbol_data is a dict with multi_timeframe key
            if isinstance(symbol_data, dict) and 'multi_timeframe' in symbol_data:
                return symbol_data['multi_timeframe']

            # Check if symbol_data is a SyncedData object directly
            if hasattr(symbol_data, 'data') and hasattr(symbol_data, 'timestamp'):
                return symbol_data

            return None

        except Exception as e:
            logger.warning(f"Failed to extract multi-timeframe data for {symbol}: {e}")
            return None

    async def initialize(self):
        """Initialize the data manager."""
        logger.info("Initializing DataManager")

        if hasattr(self.data_fetcher, 'initialize'):
            await self.data_fetcher.initialize()

        if self.timeframe_manager and hasattr(self.timeframe_manager, 'initialize'):
            await self.timeframe_manager.initialize()

        logger.info("DataManager initialization complete")

    async def shutdown(self):
        """Shutdown the data manager."""
        logger.info("Shutting down DataManager")

        if hasattr(self.data_fetcher, 'shutdown'):
            await self.data_fetcher.shutdown()

        if self.timeframe_manager and hasattr(self.timeframe_manager, 'shutdown'):
            await self.timeframe_manager.shutdown()

        logger.info("DataManager shutdown complete")
