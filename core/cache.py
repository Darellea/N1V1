"""
Redis cache implementation for trading bot data fetching.
Provides caching with TTL support and batch operations.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Cache configuration with TTL values per data type
@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # TTL values in seconds for different data types
    ttl_config: Dict[str, int] = None
    
    def __post_init__(self):
        if self.ttl_config is None:
            self.ttl_config = {
                "market_ticker": 2,      # 2 seconds
                "ohlcv": 60,             # 60 seconds
                "account_balance": 5,    # 5 seconds
                "order_book": 5,         # 5 seconds
                "trades": 10,            # 10 seconds
                "klines": 30,            # 30 seconds
                "funding_rate": 60,      # 60 seconds
                "mark_price": 10,        # 10 seconds
                "default": 30            # 30 seconds fallback
            }

class RedisCache:
    """Redis-based cache implementation with TTL support."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize Redis cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self._redis_client = None
        self._connected = False
        
    async def initialize(self) -> bool:
        """
        Initialize Redis connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import redis conditionally to avoid hard dependency
            import redis.asyncio as redis
            
            # Create Redis connection
            self._redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False  # Store as bytes for performance
            )
            
            # Test connection
            await self._redis_client.ping()
            self._connected = True
            logger.info(f"Redis cache connected to {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self._connected = False
            return False
    
    def _get_cache_key(self, data_type: str, symbol: str, **kwargs) -> str:
        """
        Generate cache key for given data type and parameters.
        
        Args:
            data_type: Type of data (e.g., 'ohlcv', 'ticker')
            symbol: Trading symbol
            **kwargs: Additional parameters for key generation
            
        Returns:
            Formatted cache key
        """
        key_parts = [data_type, symbol]
        
        # Add additional parameters based on data type
        if data_type == "ohlcv" and "timeframe" in kwargs:
            key_parts.append(kwargs["timeframe"])
        elif data_type == "klines" and "interval" in kwargs:
            key_parts.append(kwargs["interval"])
        elif data_type == "order_book" and "limit" in kwargs:
            key_parts.append(str(kwargs["limit"]))
            
        return ":".join(key_parts)
    
    def _get_ttl(self, data_type: str) -> int:
        """
        Get TTL for data type.
        
        Args:
            data_type: Type of data
            
        Returns:
            TTL in seconds
        """
        return self.config.ttl_config.get(data_type, self.config.ttl_config["default"])
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self._connected or not self._redis_client:
            return None
            
        try:
            value = await self._redis_client.get(key)
            if value is None:
                return None
                
            # Deserialize JSON
            return json.loads(value.decode('utf-8'))
            
        except Exception as e:
            logger.debug(f"Cache get failed for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or not self._redis_client:
            return False
            
        try:
            # Serialize to JSON
            serialized_value = json.dumps(value).encode('utf-8')
            
            # Set with TTL
            if ttl is None:
                ttl = self._get_ttl("default")
                
            await self._redis_client.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.debug(f"Cache set failed for key {key}: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or not self._redis_client:
            return False
            
        try:
            result = await self._redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.debug(f"Cache delete failed for key {key}: {str(e)}")
            return False
    
    async def get_market_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cached market ticker data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Ticker data or None if not cached
        """
        key = self._get_cache_key("market_ticker", symbol)
        return await self.get(key)
    
    async def set_market_ticker(self, symbol: str, ticker_data: Dict[str, Any]) -> bool:
        """
        Cache market ticker data.
        
        Args:
            symbol: Trading symbol
            ticker_data: Ticker data to cache
            
        Returns:
            True if successful, False otherwise
        """
        key = self._get_cache_key("market_ticker", symbol)
        return await self.set(key, ticker_data, ttl=self._get_ttl("market_ticker"))
    
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h") -> Optional[List[List[Any]]]:
        """
        Get cached OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            
        Returns:
            OHLCV data or None if not cached
        """
        key = self._get_cache_key("ohlcv", symbol, timeframe=timeframe)
        return await self.get(key)
    
    async def set_ohlcv(self, symbol: str, timeframe: str, ohlcv_data: List[List[Any]]) -> bool:
        """
        Cache OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            ohlcv_data: OHLCV data to cache
            
        Returns:
            True if successful, False otherwise
        """
        key = self._get_cache_key("ohlcv", symbol, timeframe=timeframe)
        return await self.set(key, ohlcv_data, ttl=self._get_ttl("ohlcv"))
    
    async def get_account_balance(self, account_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get cached account balance.
        
        Args:
            account_id: Account identifier
            
        Returns:
            Balance data or None if not cached
        """
        key = self._get_cache_key("account_balance", account_id)
        return await self.get(key)
    
    async def set_account_balance(self, account_id: str, balance_data: Dict[str, Any]) -> bool:
        """
        Cache account balance.
        
        Args:
            account_id: Account identifier
            balance_data: Balance data to cache
            
        Returns:
            True if successful, False otherwise
        """
        key = self._get_cache_key("account_balance", account_id)
        return await self.set(key, balance_data, ttl=self._get_ttl("account_balance"))
    
    async def get_multiple_ohlcv(self, symbols: List[str], timeframe: str = "1h") -> Dict[str, List[List[Any]]]:
        """
        Get cached OHLCV data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
            
        Returns:
            Dictionary mapping symbols to OHLCV data
        """
        results = {}
        
        # Try to get all cached data
        cache_keys = [self._get_cache_key("ohlcv", symbol, timeframe=timeframe) for symbol in symbols]
        cached_values = await self._redis_client.mget(cache_keys) if self._connected else [None] * len(cache_keys)
        
        # Process results
        for symbol, key, cached_value in zip(symbols, cache_keys, cached_values):
            if cached_value is not None:
                try:
                    results[symbol] = json.loads(cached_value.decode('utf-8'))
                except Exception:
                    logger.debug(f"Failed to deserialize cached OHLCV for {symbol}")
            else:
                results[symbol] = None
                
        return results
    
    async def set_multiple_ohlcv(self, data: Dict[str, List[List[Any]]], timeframe: str) -> Dict[str, bool]:
        """
        Cache OHLCV data for multiple symbols.
        
        Args:
            data: Dictionary mapping symbols to OHLCV data
            timeframe: Timeframe
            
        Returns:
            Dictionary mapping symbols to success status
        """
        results = {}
        
        # Prepare cache operations
        pipe = self._redis_client.pipeline() if self._connected else None
        operations = []
        
        for symbol, ohlcv_data in data.items():
            key = self._get_cache_key("ohlcv", symbol, timeframe=timeframe)
            serialized_value = json.dumps(ohlcv_data).encode('utf-8')
            operations.append((key, serialized_value, self._get_ttl("ohlcv")))
            
        # Execute batch operations
        if pipe and operations:
            try:
                for key, value, ttl in operations:
                    pipe.setex(key, ttl, value)
                await pipe.execute()
                
                # Mark all as successful
                for symbol in data.keys():
                    results[symbol] = True
                    
            except Exception as e:
                logger.error(f"Batch OHLCV cache set failed: {str(e)}")
                # Mark all as failed
                for symbol in data.keys():
                    results[symbol] = False
        else:
            # Fallback to individual operations
            for symbol, ohlcv_data in data.items():
                key = self._get_cache_key("ohlcv", symbol, timeframe=timeframe)
                results[symbol] = await self.set(key, ohlcv_data, ttl=self._get_ttl("ohlcv"))
                
        return results
    
    async def invalidate_symbol_data(self, symbol: str, data_types: Optional[List[str]] = None) -> int:
        """
        Invalidate all cached data for a symbol.
        
        Args:
            symbol: Trading symbol
            data_types: Specific data types to invalidate (None for all)
            
        Returns:
            Number of keys invalidated
        """
        if not self._connected or not self._redis_client:
            return 0
            
        try:
            # Get all keys for the symbol
            pattern = f"*:{symbol}:*"
            keys = await self._redis_client.keys(pattern)
            
            if not keys:
                return 0
                
            # Filter by data type if specified
            if data_types:
                filtered_keys = []
                for key in keys:
                    key_parts = key.split(":")
                    if len(key_parts) >= 2 and key_parts[0] in data_types:
                        filtered_keys.append(key)
                keys = filtered_keys
            
            # Delete keys
            if keys:
                deleted = await self._redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries for symbol {symbol}")
                return deleted
                
            return 0
            
        except Exception as e:
            logger.error(f"Cache invalidation failed for symbol {symbol}: {str(e)}")
            return 0
    
    async def clear_all(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._connected or not self._redis_client:
            return False
            
        try:
            await self._redis_client.flushdb()
            logger.info("Cleared all cache data")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                logger.info("Redis cache connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
            finally:
                self._redis_client = None
                self._connected = False

# Global cache instance
_cache_instance: Optional[RedisCache] = None
_cache_config: Optional[CacheConfig] = None

def get_cache() -> Optional[RedisCache]:
    """Get the global cache instance."""
    return _cache_instance

def initialize_cache(config: Dict[str, Any]) -> bool:
    """
    Initialize the global cache instance.
    
    Args:
        config: Cache configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    global _cache_instance, _cache_config
    
    try:
        # Create cache config
        cache_config = CacheConfig(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("db", 0),
            password=config.get("password"),
            socket_timeout=config.get("socket_timeout", 5.0),
            socket_connect_timeout=config.get("socket_connect_timeout", 5.0)
        )
        
        # Override TTL config if provided
        if "ttl_config" in config:
            cache_config.ttl_config.update(config["ttl_config"])
        
        # Create and initialize cache
        _cache_instance = RedisCache(cache_config)
        _cache_config = cache_config
        
        # Initialize connection
        return _cache_instance.initialize()
        
    except Exception as e:
        logger.error(f"Failed to initialize cache: {str(e)}")
        return False

def close_cache() -> None:
    """Close the global cache instance."""
    global _cache_instance
    
    if _cache_instance:
        asyncio.create_task(_cache_instance.close())
        _cache_instance = None

# Import asyncio for async operations
import asyncio
