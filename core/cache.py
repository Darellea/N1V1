"""
Redis cache implementation for trading bot data fetching.
Provides caching with TTL support and batch operations.
"""

import json
import logging
import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import heapq
import psutil
import os

from .logging_utils import get_structured_logger
from .logging_utils import LogSensitivity

logger = get_structured_logger("cache", LogSensitivity.INFO)

# Import configuration from centralized system
from .config_manager import get_config_manager
from .interfaces import CacheConfig, MemoryConfig, EvictionPolicy

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
            logger.info(f"Redis cache connected to {self.config.host}:{self.config.port}", component="cache", operation="initialize", host=self.config.host, port=self.config.port)
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
                logger.info("Redis cache connection closed", component="cache", operation="close")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}", component="cache", operation="close", error=str(e))
            finally:
                self._redis_client = None
                self._connected = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # Memory monitoring and eviction methods
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception as e:
            logger.debug(f"Failed to get memory usage: {str(e)}")
            return 0.0

    def _check_memory_thresholds(self) -> Dict[str, bool]:
        """Check if memory thresholds are exceeded."""
        memory_mb = self._get_memory_usage()
        config = self.config.memory_config

        return {
            "exceeds_max": memory_mb >= config.max_memory_mb,
            "exceeds_warning": memory_mb >= config.warning_memory_mb,
            "exceeds_cleanup": memory_mb >= config.cleanup_memory_mb
        }

    async def _evict_expired_entries(self) -> int:
        """Evict expired cache entries based on TTL."""
        if not self._connected or not self._redis_client:
            return 0

        try:
            # Get all keys (this is a simplified approach for Redis)
            # In production, you might want to use Redis SCAN for large keysets
            all_keys = await self._redis_client.keys("*")
            if not all_keys:
                return 0

            evicted = 0
            batch_size = self.config.memory_config.eviction_batch_size

            # Check TTL for each key and evict if expired
            for i in range(0, len(all_keys), batch_size):
                batch_keys = all_keys[i:i + batch_size]
                ttls = await self._redis_client.ttl(batch_keys)

                keys_to_delete = []
                for key, ttl in zip(batch_keys, ttls):
                    if ttl == -2:  # Key doesn't exist
                        continue
                    elif ttl == -1:  # Key has no TTL
                        continue
                    elif ttl == 0:  # Key is expired
                        keys_to_delete.append(key)

                if keys_to_delete:
                    deleted = await self._redis_client.delete(*keys_to_delete)
                    evicted += deleted

            if evicted > 0:
                logger.info(f"Evicted {evicted} expired cache entries")

            return evicted

        except Exception as e:
            logger.error(f"Failed to evict expired entries: {str(e)}")
            return 0

    async def _enforce_cache_limits(self) -> int:
        """Enforce cache size limits using configured eviction policy."""
        if not self._connected or not self._redis_client:
            return 0

        try:
            # Get current cache size
            cache_size = await self._redis_client.dbsize()

            if cache_size <= self.config.max_cache_size:
                return 0  # No eviction needed

            # Calculate how many entries to evict
            entries_to_evict = min(
                self.config.memory_config.eviction_batch_size,
                cache_size - self.config.max_cache_size
            )

            evicted = 0

            if self.config.eviction_policy == EvictionPolicy.LRU:
                evicted = await self._evict_lru(entries_to_evict)
            elif self.config.eviction_policy == EvictionPolicy.LFU:
                evicted = await self._evict_lfu(entries_to_evict)
            else:  # TTL-based eviction (default)
                evicted = await self._evict_oldest(entries_to_evict)

            if evicted > 0:
                logger.info(f"Evicted {evicted} entries using {self.config.eviction_policy.value} policy")

            return evicted

        except Exception as e:
            logger.error(f"Failed to enforce cache limits: {str(e)}")
            return 0

    async def _evict_lru(self, count: int) -> int:
        """Evict least recently used entries."""
        if not self._connected or not self._redis_client:
            return 0

        try:
            # Get all keys and their idle times
            all_keys = await self._redis_client.keys("*")
            if not all_keys:
                return 0

            # Get idle times for all keys
            idle_times = await self._redis_client.object("idletime", all_keys)

            # Create list of (idle_time, key) tuples
            key_idle_pairs = list(zip(idle_times, all_keys))

            # Sort by idle time (descending - most idle first)
            key_idle_pairs.sort(key=lambda x: x[0], reverse=True)

            # Get keys to evict
            keys_to_evict = [key for _, key in key_idle_pairs[:count]]

            if keys_to_evict:
                deleted = await self._redis_client.delete(*keys_to_evict)
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Failed to evict LRU entries: {str(e)}")
            return 0

    async def _evict_lfu(self, count: int) -> int:
        """Evict least frequently used entries."""
        if not self._connected or not self._redis_client:
            return 0

        try:
            # Get all keys and their access frequencies
            all_keys = await self._redis_client.keys("*")
            if not all_keys:
                return 0

            # Get access frequencies for all keys
            access_freqs = await self._redis_client.object("freq", all_keys)

            # Create list of (access_freq, key) tuples
            key_freq_pairs = list(zip(access_freqs, all_keys))

            # Sort by access frequency (ascending - least used first)
            key_freq_pairs.sort(key=lambda x: x[0])

            # Get keys to evict
            keys_to_evict = [key for _, key in key_freq_pairs[:count]]

            if keys_to_evict:
                deleted = await self._redis_client.delete(*keys_to_evict)
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Failed to evict LFU entries: {str(e)}")
            return 0

    async def _evict_oldest(self, count: int) -> int:
        """Evict oldest entries (TTL-based)."""
        if not self._connected or not self._redis_client:
            return 0

        try:
            # Get all keys and their TTL values
            all_keys = await self._redis_client.keys("*")
            if not all_keys:
                return 0

            # Get TTL for all keys
            ttls = await self._redis_client.ttl(all_keys)

            # Create list of (ttl, key) tuples
            key_ttl_pairs = list(zip(ttls, all_keys))

            # Sort by TTL (ascending - shortest TTL first)
            key_ttl_pairs.sort(key=lambda x: x[0])

            # Get keys to evict (prefer shorter TTL)
            keys_to_evict = [key for ttl, key in key_ttl_pairs[:count] if ttl > 0]

            if keys_to_evict:
                deleted = await self._redis_client.delete(*keys_to_evict)
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Failed to evict oldest entries: {str(e)}")
            return 0

    async def perform_maintenance(self) -> Dict[str, Any]:
        """Perform cache maintenance including eviction and memory monitoring."""
        if not self._connected or not self._redis_client:
            return {"error": "Cache not connected"}

        try:
            # Check memory thresholds
            memory_status = self._check_memory_thresholds()
            memory_mb = self._get_memory_usage()

            # Get cache statistics
            cache_size = await self._redis_client.dbsize()

            maintenance_results = {
                "memory_mb": memory_mb,
                "cache_size": cache_size,
                "memory_status": memory_status,
                "evicted_expired": 0,
                "evicted_limits": 0,
                "maintenance_performed": False
            }

            # Perform maintenance if needed
            if (memory_status["exceeds_cleanup"] or
                cache_size > self.config.max_cache_size):

                maintenance_results["maintenance_performed"] = True

                # Evict expired entries
                maintenance_results["evicted_expired"] = await self._evict_expired_entries()

                # Enforce cache limits
                maintenance_results["evicted_limits"] = await self._enforce_cache_limits()

                # Log memory warnings
                if memory_status["exceeds_warning"]:
                    logger.warning(".2f")
                if memory_status["exceeds_max"]:
                    logger.error(".2f")

            return maintenance_results

        except Exception as e:
            logger.error(f"Cache maintenance failed: {str(e)}")
            return {"error": str(e)}

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self._connected or not self._redis_client:
            return {"error": "Cache not connected"}

        try:
            cache_size = await self._redis_client.dbsize()
            memory_mb = self._get_memory_usage()
            memory_status = self._check_memory_thresholds()

            # Get info about Redis memory usage
            info = await self._redis_client.info("memory")

            return {
                "cache_size": cache_size,
                "max_cache_size": self.config.max_cache_size,
                "memory_mb": memory_mb,
                "memory_status": memory_status,
                "eviction_policy": self.config.eviction_policy.value,
                "redis_memory_used": info.get("used_memory", 0) / 1024 / 1024,
                "redis_memory_peak": info.get("used_memory_peak", 0) / 1024 / 1024,
                "thresholds": {
                    "max_memory_mb": self.config.memory_config.max_memory_mb,
                    "warning_memory_mb": self.config.memory_config.warning_memory_mb,
                    "cleanup_memory_mb": self.config.memory_config.cleanup_memory_mb
                }
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {"error": str(e)}

# Global cache instance with thread-safe access
_cache_instance: Optional[RedisCache] = None
_cache_config: Optional[CacheConfig] = None
_cache_lock = threading.RLock()  # Reentrant lock for thread safety
_cache_cleanup_registered = False

def get_cache() -> Optional[RedisCache]:
    """Get the global cache instance with thread-safe access."""
    with _cache_lock:
        return _cache_instance

def initialize_cache(config: Dict[str, Any]) -> bool:
    """
    Initialize the global cache instance with thread-safe access.

    Args:
        config: Cache configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    global _cache_instance, _cache_config, _cache_cleanup_registered

    with _cache_lock:
        try:
            # Check if already initialized
            if _cache_instance is not None:
                logger.warning("Cache already initialized, skipping re-initialization")
                return True

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

            # Configure eviction policy
            if "eviction_policy" in config:
                policy_str = config["eviction_policy"].lower()
                if policy_str == "lru":
                    cache_config.eviction_policy = EvictionPolicy.LRU
                elif policy_str == "lfu":
                    cache_config.eviction_policy = EvictionPolicy.LFU
                else:  # Default to TTL
                    cache_config.eviction_policy = EvictionPolicy.TTL

            # Configure cache size limit
            if "max_cache_size" in config:
                cache_config.max_cache_size = config["max_cache_size"]

            # Configure memory monitoring
            if "memory_config" in config:
                mem_config = config["memory_config"]
                if isinstance(mem_config, dict):
                    cache_config.memory_config = MemoryConfig(
                        max_memory_mb=mem_config.get("max_memory_mb", 500.0),
                        warning_memory_mb=mem_config.get("warning_memory_mb", 400.0),
                        cleanup_memory_mb=mem_config.get("cleanup_memory_mb", 350.0),
                        eviction_batch_size=mem_config.get("eviction_batch_size", 100),
                        memory_check_interval=mem_config.get("memory_check_interval", 60.0)
                    )

            # Create and initialize cache
            _cache_instance = RedisCache(cache_config)
            _cache_config = cache_config

            # Initialize connection synchronously for thread safety
            try:
                # Use asyncio.run() for synchronous initialization
                success = asyncio.run(_cache_instance.initialize())
                if success and not _cache_cleanup_registered:
                    # Register cleanup handler for application exit
                    import atexit
                    atexit.register(_cleanup_cache_on_exit)
                    _cache_cleanup_registered = True
                    logger.info("Cache cleanup handler registered with atexit", component="cache", operation="initialize")
                return success
            except RuntimeError:
                # If already in an event loop, initialize asynchronously
                logger.warning("Already in event loop, initializing cache asynchronously")
                asyncio.create_task(_cache_instance.initialize())
                if not _cache_cleanup_registered:
                    # Register cleanup handler for application exit
                    import atexit
                    atexit.register(_cleanup_cache_on_exit)
                    _cache_cleanup_registered = True
                    logger.info("Cache cleanup handler registered with atexit", component="cache", operation="initialize")
                return True

        except Exception as e:
            logger.error(f"Failed to initialize cache: {str(e)}", component="cache", operation="initialize", error=str(e))
            return False

def close_cache() -> None:
    """Close the global cache instance with thread-safe access."""
    global _cache_instance

    with _cache_lock:
        if _cache_instance:
            try:
                # Use asyncio.run() for synchronous closing
                asyncio.run(_cache_instance.close())
                logger.info("Cache closed successfully", component="cache", operation="close")
            except RuntimeError:
                # If already in an event loop, close asynchronously
                asyncio.create_task(_cache_instance.close())
                logger.info("Cache close initiated asynchronously", component="cache", operation="close")
            except Exception as e:
                logger.error(f"Error closing cache: {str(e)}", component="cache", operation="close", error=str(e))
            finally:
                _cache_instance = None

def _cleanup_cache_on_exit() -> None:
    """Cleanup function called on application exit."""
    logger.info("Performing cache cleanup on application exit", component="cache", operation="cleanup")
    try:
        close_cache()
        logger.info("Cache cleanup completed successfully", component="cache", operation="cleanup")
    except Exception as e:
        logger.error(f"Error during cache cleanup on exit: {str(e)}", component="cache", operation="cleanup", error=str(e))

# Context manager for global cache instance
class CacheContext:
    """Context manager for safe cache operations with automatic cleanup."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config
        self._cache_initialized = False

    async def __aenter__(self) -> RedisCache:
        """Enter context and ensure cache is initialized."""
        if not _cache_instance and self.config:
            self._cache_initialized = initialize_cache(self.config)
            if not self._cache_initialized:
                raise RuntimeError("Failed to initialize cache in context manager")

        if not _cache_instance:
            raise RuntimeError("Cache not initialized and no config provided")

        return _cache_instance

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context - cleanup will be handled by atexit or explicit close."""
        # Cache cleanup is handled by atexit or explicit close_cache() calls
        # Individual operations don't need cleanup here as Redis connection is persistent
        pass

# Import asyncio for async operations
import asyncio
