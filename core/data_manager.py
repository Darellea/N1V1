"""
DataManager - Handles market data fetching and caching operations.

Manages data fetching from exchanges, caching, multi-timeframe data,
and provides a unified interface for market data access.
"""

import asyncio
import time
from typing import Any, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

from .interfaces import DataManagerInterface
from .logging_utils import LogSensitivity, get_structured_logger
from .utils.error_utils import (
    CircuitBreaker,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorSeverity,
)
from .exceptions import SchemaValidationError
from api.models import validate_ticker_data, validate_market_data

logger = get_structured_logger("core.data_manager", LogSensitivity.SECURE)
error_handler = ErrorHandler("data_manager")


class DataManager(DataManagerInterface):
    """
    Manages market data fetching, caching, and processing.

    Responsibilities:
    - Coordinate data fetching from exchanges
    - Manage caching for performance
    - Handle multi-timeframe data synchronization
    - Provide unified data access interface
    - Handle data validation and error recovery
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize the DataManager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.mode = config.get("environment", {}).get("mode", "paper")
        self.portfolio_mode = bool(
            config.get("trading", {}).get("portfolio_mode", False)
        )
        self.pairs = []

        # Component references
        self.data_fetcher = None
        self.timeframe_manager = None

        # Import configuration from centralized system
        from .config_manager import get_config_manager

        config_manager = get_config_manager()
        dm_config = config_manager.get_data_manager_config()

        # Caching configuration from centralized config
        self.cache_enabled = dm_config.cache_enabled
        self.cache_ttl = dm_config.cache_ttl
        self.market_data_cache = {}
        self.cache_timestamp = 0.0

        # Circuit breaker for external service calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("circuit_breaker", {}).get(
                "failure_threshold", 5
            ),
            recovery_timeout=config.get("circuit_breaker", {}).get(
                "recovery_timeout", 60.0
            ),
        )

    def set_components(self, data_fetcher, timeframe_manager=None):
        """Set component references."""
        self.data_fetcher = data_fetcher
        self.timeframe_manager = timeframe_manager

    def set_trading_pairs(self, pairs: list[str]):
        """Set the trading pairs for data management."""
        self.pairs = pairs
        logger.info(f"DataManager configured for {len(pairs)} pairs: {pairs}")

    async def fetch_market_data(self) -> dict[str, Any]:
        """Fetch market data with caching and multi-timeframe support."""
        current_time = time.time()

        # Check if we should use cached data
        if self._should_use_cache(current_time):
            logger.debug("Using cached market data")
            return self.market_data_cache

        try:
            # Fetch fresh market data
            market_data = await self._fetch_fresh_market_data()

            # Fetch multi-timeframe data if available
            if self.timeframe_manager and self.pairs:
                multi_timeframe_data = await self._fetch_multi_timeframe_data()
                market_data = self._combine_market_data(
                    market_data, multi_timeframe_data
                )

            # Cache the fetched data
            self._cache_market_data(market_data, current_time)

            return market_data

        except (ConnectionError, TimeoutError) as e:
            context = ErrorContext(
                component="data_manager",
                operation="fetch_market_data",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.NETWORK,
                metadata={"fetch_error": str(e)},
            )
            await error_handler.handle_error(e, context)
        except aiohttp.ClientError as e:
            if aiohttp:
                context = ErrorContext(
                    component="data_manager",
                    operation="fetch_market_data",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.NETWORK,
                    metadata={"fetch_error": str(e)},
                )
                await error_handler.handle_error(e, context)
            else:
                raise
        except (ValueError, TypeError, KeyError) as e:
            context = ErrorContext(
                component="data_manager",
                operation="fetch_market_data",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.DATA,
                metadata={"fetch_error": str(e)},
            )
            await error_handler.handle_error(e, context)
        except Exception as e:
            context = ErrorContext(
                component="data_manager",
                operation="fetch_market_data",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.DATA,
                metadata={"fetch_error": str(e)},
            )
            await error_handler.handle_error(e, context)

        # Return cached data if available, otherwise empty dict
        if self.market_data_cache:
            logger.warning("Using stale cached data due to fetch failure")
            return self.market_data_cache

        return {}

    def _should_use_cache(self, current_time: float) -> bool:
        """Determine if cached data should be used."""
        if not self.cache_enabled:
            return False

        if not self.market_data_cache:
            return False

        time_since_cache = current_time - self.cache_timestamp
        return time_since_cache < self.cache_ttl

    async def _fetch_fresh_market_data(self) -> dict[str, Any]:
        """Fetch fresh market data from the exchange."""
        if not self.data_fetcher:
            raise RuntimeError("DataFetcher not set")

        try:
            if self.portfolio_mode and hasattr(self.data_fetcher, "get_realtime_data"):
                return await self._fetch_portfolio_realtime_data()
            elif not self.portfolio_mode and hasattr(
                self.data_fetcher, "get_historical_data"
            ):
                return await self._fetch_single_historical_data()
            elif hasattr(self.data_fetcher, "get_multiple_historical_data"):
                return await self._fetch_multiple_historical_data()
            else:
                logger.warning("No suitable data fetching method available")
                return {}

        except Exception:
            logger.exception("Failed to fetch fresh market data")
            raise

    async def _fetch_portfolio_realtime_data(self) -> dict[str, Any]:
        """Fetch portfolio realtime data with caching support."""
        if not self.pairs:
            return {}

        try:
            # Use circuit breaker for external service calls
            raw_data = await self.circuit_breaker.call(
                self.data_fetcher.get_realtime_data, self.pairs
            )

            # Validate and parse the data
            validated_data = {}
            for symbol, data in raw_data.items():
                if isinstance(data, dict):
                    # Try to validate as ticker data
                    try:
                        validated_ticker = validate_ticker_data(data)
                        validated_data[symbol] = validated_ticker.model_dump()
                    except SchemaValidationError:
                        # If ticker validation fails, try general market data validation
                        try:
                            validated_market = validate_market_data(
                                {
                                    "symbol": symbol,
                                    "data_type": "realtime",
                                    "payload": data,
                                    "timestamp": time.time(),
                                }
                            )
                            validated_data[symbol] = validated_market.model_dump()
                        except SchemaValidationError as e:
                            logger.error(
                                f"Schema validation failed for {symbol} realtime data: {e}"
                            )
                            # Continue with raw data but log the issue
                            validated_data[symbol] = data
                else:
                    # Non-dict data (e.g., DataFrames) pass through
                    validated_data[symbol] = data

            return validated_data

        except Exception:
            logger.exception("Failed to fetch portfolio realtime data")
            raise

    async def _fetch_single_historical_data(self) -> dict[str, Any]:
        """Fetch single historical data."""
        if not self.pairs:
            return {}

        symbol = self.pairs[0]
        timeframe = self.config.get("backtesting", {}).get("timeframe", "1h")

        try:
            # Use circuit breaker for external service calls
            df = await self.circuit_breaker.call(
                self.data_fetcher.get_historical_data,
                symbol=symbol,
                timeframe=timeframe,
                limit=100,
            )
            return {symbol: df}
        except Exception:
            logger.exception(f"Failed to fetch historical data for {symbol}")
            raise

    async def _fetch_multiple_historical_data(self) -> dict[str, Any]:
        """Fetch multiple historical data."""
        if not self.pairs:
            return {}

        try:
            # Use circuit breaker for external service calls
            return await self.circuit_breaker.call(
                self.data_fetcher.get_multiple_historical_data, self.pairs
            )
        except Exception:
            logger.exception("Failed to fetch multiple historical data")
            raise

    async def _fetch_multi_timeframe_data(self) -> dict[str, Any]:
        """Fetch multi-timeframe data for all symbols."""
        multi_timeframe_data = {}

        for symbol in self.pairs:
            try:
                synced_data = await self.timeframe_manager.fetch_multi_timeframe_data(
                    symbol
                )
                if synced_data:
                    multi_timeframe_data[symbol] = synced_data
                    logger.debug(f"Fetched multi-timeframe data for {symbol}")
                else:
                    logger.warning(f"Failed to fetch multi-timeframe data for {symbol}")
            except Exception as e:
                logger.warning(f"Error fetching multi-timeframe data for {symbol}: {e}")
                continue

        return multi_timeframe_data

    def _combine_market_data(
        self, market_data: dict[str, Any], multi_timeframe_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Combine single-timeframe and multi-timeframe market data."""
        combined_data = market_data.copy()

        for symbol, synced_data in multi_timeframe_data.items():
            if symbol in combined_data:
                # Add multi-timeframe data to existing symbol data
                combined_data[symbol] = {
                    "single_timeframe": combined_data[symbol],
                    "multi_timeframe": synced_data,
                }
            else:
                # Only multi-timeframe data available
                combined_data[symbol] = {"multi_timeframe": synced_data}

        return combined_data

    def _cache_market_data(
        self, combined_data: dict[str, Any], current_time: float
    ) -> None:
        """Cache the fetched market data."""
        self.market_data_cache = combined_data
        self.cache_timestamp = current_time
        logger.debug("Fetched and cached market data")

    async def get_symbol_data(
        self, symbol: str, timeframe: str = None
    ) -> Optional[Any]:
        """Get data for a specific symbol and timeframe."""
        try:
            if not self.market_data_cache:
                await self.fetch_market_data()

            if symbol not in self.market_data_cache:
                logger.warning(f"Symbol {symbol} not found in cached data")
                return None

            symbol_data = self.market_data_cache[symbol]

            # If timeframe specified, try to get specific timeframe data
            if timeframe and isinstance(symbol_data, dict):
                if "multi_timeframe" in symbol_data:
                    mt_data = symbol_data["multi_timeframe"]
                    if hasattr(mt_data, "get_timeframe_data"):
                        return mt_data.get_timeframe_data(timeframe)
                elif "single_timeframe" in symbol_data:
                    return symbol_data["single_timeframe"]

            return symbol_data

        except (KeyError, TypeError, AttributeError) as e:
            logger.error(
                f"Failed to get symbol data for {symbol} - data structure error: {e}"
            )
            return None
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout getting symbol data for {symbol}: {e}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error getting symbol data for {symbol}: {e}")
            return None

    def clear_cache(self):
        """Clear the market data cache."""
        self.market_data_cache = {}
        self.cache_timestamp = 0.0
        logger.info("Market data cache cleared")

    def get_cache_status(self) -> dict[str, Any]:
        """Get cache status information."""
        return {
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "cache_size": len(self.market_data_cache),
            "cache_age": time.time() - self.cache_timestamp
            if self.cache_timestamp > 0
            else None,
            "cached_symbols": list(self.market_data_cache.keys())
            if self.market_data_cache
            else [],
        }

    async def initialize(self) -> None:
        """Initialize the data manager and its components."""
        try:
            if self.data_fetcher and hasattr(self.data_fetcher, "initialize"):
                await self.data_fetcher.initialize()

            if self.timeframe_manager and hasattr(self.timeframe_manager, "initialize"):
                await self.timeframe_manager.initialize()

            logger.info("DataManager initialized successfully")

        except Exception as e:
            context = ErrorContext(
                component="data_manager",
                operation="initialize",
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.DATA,
                metadata={"init_error": str(e)},
            )
            await error_handler.handle_error(e, context)
            raise

    async def shutdown(self) -> None:
        """Shutdown the data manager and its components."""
        try:
            if self.data_fetcher and hasattr(self.data_fetcher, "shutdown"):
                await self.data_fetcher.shutdown()

            if self.timeframe_manager and hasattr(self.timeframe_manager, "shutdown"):
                await self.timeframe_manager.shutdown()

            self.clear_cache()
            logger.info("DataManager shutdown complete")

        except (AttributeError, TypeError) as e:
            logger.error(f"Error during DataManager shutdown - component error: {e}")
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout during DataManager shutdown: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during DataManager shutdown: {e}")

    def get_data_manager_status(self) -> dict[str, Any]:
        """Get data manager status information."""
        return {
            "mode": self.mode,
            "portfolio_mode": self.portfolio_mode,
            "trading_pairs": self.pairs,
            "cache_status": self.get_cache_status(),
            "components_initialized": {
                "data_fetcher": self.data_fetcher is not None,
                "timeframe_manager": self.timeframe_manager is not None,
            },
        }
