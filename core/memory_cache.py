"""
core/memory_cache.py

In-memory LRU cache with TTL support and memory caps.
Provides bounded caching to prevent memory exhaustion in long-running applications.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cache entry with value, timestamp, and access tracking."""

    value: Any
    timestamp: float
    ttl_seconds: Optional[float] = None  # Per-entry TTL override
    access_count: int = 0
    last_access: float = field(init=False)

    def __post_init__(self):
        self.last_access = self.timestamp


@dataclass
class CacheMetrics:
    """Cache performance and usage metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    sets: int = 0
    deletes: int = 0
    current_size: int = 0
    max_size: int = 0
    total_accesses: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        return self.hits / self.total_accesses if self.total_accesses > 0 else 0.0


class Cache:
    """
    LRU (Least Recently Used) cache with TTL (Time To Live) support.

    Features:
    - Automatic eviction of least recently used items when capacity exceeded
    - TTL-based expiration of entries
    - Thread-safe operations
    - Comprehensive metrics tracking
    - Configurable eviction policies
    """

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_seconds: int = 3600,
        enable_metrics: bool = True,
    ):
        """
        Initialize the LRU cache.

        Args:
            max_entries: Maximum number of entries to store
            ttl_seconds: Time-to-live for entries in seconds (None = no TTL)
            enable_metrics: Whether to track performance metrics
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.enable_metrics = enable_metrics

        # Thread-safe storage using OrderedDict for LRU ordering
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Metrics tracking
        self._metrics = CacheMetrics() if enable_metrics else None

        logger.info(
            f"Cache initialized: max_entries={max_entries}, "
            f"ttl_seconds={ttl_seconds}, metrics={enable_metrics}"
        )

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                if self._metrics:
                    self._metrics.misses += 1
                    self._metrics.total_accesses += 1
                return None

            entry = self._cache[key]

            # Check TTL expiration
            if self._is_expired(entry):
                # Remove expired entry
                del self._cache[key]
                if self._metrics:
                    self._metrics.evictions += 1
                    self._metrics.misses += 1
                    self._metrics.total_accesses += 1
                    self._metrics.current_size = len(self._cache)
                return None

            # Update access tracking
            entry.access_count += 1
            entry.last_access = time.time()

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            if self._metrics:
                self._metrics.hits += 1
                self._metrics.total_accesses += 1

            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store
            ttl_seconds: Optional TTL override for this entry
        """
        with self._lock:
            current_time = time.time()
            entry_ttl = ttl_seconds if ttl_seconds is not None else self.ttl_seconds

            # Create or update entry
            if key in self._cache:
                # Update existing entry
                self._cache[key].value = value
                self._cache[key].timestamp = current_time
                self._cache[key].ttl_seconds = entry_ttl  # Store per-entry TTL
                self._cache.move_to_end(key)
            else:
                # Create new entry
                entry = CacheEntry(
                    value=value, timestamp=current_time, ttl_seconds=entry_ttl
                )
                self._cache[key] = entry

                # Prune expired or over-capacity entries
                self.cleanup_expired()
                if len(self._cache) > self.max_entries:
                    self._evict_oldest()

            if self._metrics:
                self._metrics.sets += 1
                self._metrics.current_size = len(self._cache)
                self._metrics.max_size = max(self._metrics.max_size, len(self._cache))

    def delete(self, key: str) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if self._metrics:
                    self._metrics.deletes += 1
                    self._metrics.current_size = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            if self._metrics:
                self._metrics.current_size = 0

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            removed_count = len(expired_keys)
            if self._metrics and removed_count > 0:
                self._metrics.evictions += removed_count
                self._metrics.current_size = len(self._cache)

            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} expired cache entries")

            return removed_count

    def get_metrics(self) -> Optional[CacheMetrics]:
        """
        Get cache performance metrics.

        Returns:
            CacheMetrics object or None if metrics disabled
        """
        return self._metrics

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            current_time = time.time()

            # Calculate TTL statistics
            ttl_stats = {}
            if self.ttl_seconds:
                ttl_stats = {
                    "ttl_seconds": self.ttl_seconds,
                    "entries_with_ttl": len(self._cache),
                }

            # Calculate access patterns
            access_stats = {}
            if self._cache:
                access_counts = [entry.access_count for entry in self._cache.values()]
                access_stats = {
                    "avg_access_count": sum(access_counts) / len(access_counts),
                    "max_access_count": max(access_counts),
                    "min_access_count": min(access_counts),
                }

            stats = {
                "current_size": len(self._cache),
                "max_entries": self.max_entries,
                "utilization_percent": (len(self._cache) / self.max_entries) * 100,
                "ttl_enabled": self.ttl_seconds is not None,
                "metrics_enabled": self.enable_metrics,
                **ttl_stats,
                **access_stats,
            }

            # Add metrics if enabled
            if self._metrics:
                stats.update(
                    {
                        "hits": self._metrics.hits,
                        "misses": self._metrics.misses,
                        "evictions": self._metrics.evictions,
                        "sets": self._metrics.sets,
                        "deletes": self._metrics.deletes,
                        "hit_rate": self._metrics.hit_rate,
                        "total_accesses": self._metrics.total_accesses,
                    }
                )

            return stats

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if entry.ttl_seconds is None:
            return False
        return (time.time() - entry.timestamp) > entry.ttl_seconds

    def _evict_oldest(self) -> None:
        """Evict the oldest item."""
        if not self._cache:
            return

        # Remove the first item (least recently used)
        evicted_key, _ = self._cache.popitem(last=False)

        if self._metrics:
            self._metrics.evictions += 1
            self._metrics.current_size = len(self._cache)

        logger.debug(f"Evicted oldest entry: {evicted_key}")

    def __len__(self) -> int:
        """Get the current number of items in the cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache (non-expired)."""
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            if self._is_expired(entry):
                del self._cache[key]
                if self._metrics:
                    self._metrics.evictions += 1
                    self._metrics.current_size = len(self._cache)
                return False
            return True

    def __repr__(self) -> str:
        """String representation of the cache."""
        with self._lock:
            return f"Cache(size={len(self._cache)}/{self.max_entries}, ttl={self.ttl_seconds})"


# Backward-compatible alias for existing code
class LRUCacheWithTTL(Cache):
    """Backward-compatible alias for Cache class."""

    def __init__(
        self, max_size: int = 1000, ttl_seconds: int = 3600, enable_metrics: bool = True
    ):
        """
        Initialize the LRU cache with backward-compatible parameter names.

        Args:
            max_size: Maximum number of entries to store (maps to max_entries)
            ttl_seconds: Time-to-live for entries in seconds
            enable_metrics: Whether to track performance metrics
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        super().__init__(
            max_entries=max_size, ttl_seconds=ttl_seconds, enable_metrics=enable_metrics
        )

    @property
    def max_size(self):
        """Backward-compatible property for max_size."""
        return self.max_entries

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics with backward-compatible keys.

        Returns:
            Dictionary with cache statistics
        """
        stats = super().get_stats()
        # Add backward-compatible key
        stats["max_size"] = stats.get("max_entries", 0)
        return stats

    def __repr__(self) -> str:
        """String representation with backward-compatible class name."""
        with self._lock:
            return f"LRUCacheWithTTL(size={len(self._cache)}/{self.max_entries}, ttl={self.ttl_seconds})"


# Global cache instances for common use cases
_default_cache = LRUCacheWithTTL(max_size=1000, ttl_seconds=300)  # 5 minutes TTL
_large_cache = LRUCacheWithTTL(max_size=10000, ttl_seconds=3600)  # 1 hour TTL
_no_ttl_cache = LRUCacheWithTTL(max_size=500, ttl_seconds=None)  # No TTL


def get_default_cache() -> LRUCacheWithTTL:
    """Get the default cache instance (1000 items, 5min TTL)."""
    return _default_cache


def get_large_cache() -> LRUCacheWithTTL:
    """Get the large cache instance (10000 items, 1hr TTL)."""
    return _large_cache


def get_no_ttl_cache() -> LRUCacheWithTTL:
    """Get the no-TTL cache instance (500 items, no expiration)."""
    return _no_ttl_cache


def create_cache(
    max_entries: int = 1000,
    ttl_seconds: Optional[float] = 300,
    enable_metrics: bool = True,
    max_size: Optional[int] = None,
) -> LRUCacheWithTTL:
    """
    Create a new cache instance with specified parameters.

    Args:
        max_entries: Maximum number of entries (deprecated, use max_size)
        ttl_seconds: TTL in seconds (None for no TTL)
        enable_metrics: Whether to enable metrics tracking
        max_size: Maximum number of entries (backward-compatible alias)

    Returns:
        New LRUCacheWithTTL instance
    """
    # Use max_size if provided, otherwise max_entries
    actual_max_size = max_size if max_size is not None else max_entries

    return LRUCacheWithTTL(
        max_size=actual_max_size, ttl_seconds=ttl_seconds, enable_metrics=enable_metrics
    )


__all__ = [
    "Cache",
    "CacheEntry",
    "CacheMetrics",
    "LRUCacheWithTTL",
    "get_default_cache",
    "get_large_cache",
    "get_no_ttl_cache",
    "create_cache",
]
