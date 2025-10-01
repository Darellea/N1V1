"""
MemoryManager - Memory management and resource cleanup component.

Handles explicit resource cleanup, object pooling, memory usage monitoring,
and automatic cleanup triggers for optimal memory usage.
"""

import gc
import logging
import os
import threading
import time
import tracemalloc
from typing import Any, Dict, List, Optional
from weakref import WeakSet, ref

import psutil

from .interfaces import MemoryManagerInterface

logger = logging.getLogger(__name__)


class MemoryManager(MemoryManagerInterface):
    """
    Comprehensive memory management system with resource cleanup,
    object pooling, and memory monitoring capabilities.
    """

    def __init__(self, enable_monitoring: bool = True, cleanup_interval: float = 300.0):
        """Initialize the MemoryManager.

        Args:
            enable_monitoring: Whether to enable memory monitoring
            cleanup_interval: Interval for automatic cleanup in seconds
        """
        # Import configuration from centralized system
        from .config_manager import get_config_manager

        config_manager = get_config_manager()
        memory_config = config_manager.get_memory_config()

        self.enable_monitoring = (
            enable_monitoring
            if enable_monitoring is not None
            else memory_config.enable_monitoring
        )
        self.cleanup_interval = (
            cleanup_interval
            if cleanup_interval != 300.0
            else memory_config.cleanup_interval
        )

        # Thread synchronization locks
        self._pools_lock = threading.RLock()  # For object pools operations
        self._tracking_lock = threading.RLock()  # For object tracking operations
        self._snapshots_lock = threading.RLock()  # For memory snapshots operations
        self._metrics_lock = threading.RLock()  # For performance metrics operations
        self._callbacks_lock = threading.RLock()  # For cleanup callbacks operations

        # Object pools
        self._object_pools: Dict[str, List[Any]] = {}
        self._pool_sizes: Dict[str, int] = {}
        self._pool_cleanup_times: Dict[str, float] = {}

        # Weak references for tracking
        self._tracked_objects: WeakSet = WeakSet()
        self._object_refs: Dict[str, ref] = {}

        # Memory monitoring with configurable thresholds
        self._memory_snapshots: List[Dict[str, Any]] = []
        self._memory_thresholds: Dict[str, float] = {
            "warning_mb": memory_config.warning_memory_mb,
            "critical_mb": memory_config.max_memory_mb,
            "cleanup_mb": memory_config.cleanup_memory_mb,
        }

        # Resource cleanup callbacks
        self._cleanup_callbacks: List[callable] = []

        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False

        # Performance metrics
        self._cleanup_count = 0
        self._memory_warnings = 0
        self._memory_criticals = 0

        # Initialize monitoring if enabled and not in testing mode
        if enable_monitoring and not os.environ.get("TESTING"):
            self._start_memory_monitoring()

    def _start_memory_monitoring(self):
        """Start the memory monitoring thread."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()

        self._monitoring_thread = threading.Thread(
            target=self._memory_monitoring_loop, daemon=True, name="MemoryMonitor"
        )
        self._monitoring_thread.start()
        logger.info("Memory monitoring started")

    def _memory_monitoring_loop(self):
        """Main memory monitoring loop."""
        while not self._stop_monitoring:
            try:
                self._check_memory_usage()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.exception(f"Error in memory monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying

    def _check_memory_usage(self):
        """Check current memory usage and trigger cleanup if needed."""
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Take memory snapshot
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")

            memory_info = {
                "timestamp": time.time(),
                "memory_mb": memory_mb,
                "top_allocations": [
                    {
                        "size_mb": stat.size / 1024 / 1024,
                        "count": stat.count,
                        "file": stat.traceback[0].filename
                        if stat.traceback
                        else "unknown",
                        "line": stat.traceback[0].lineno if stat.traceback else 0,
                    }
                    for stat in top_stats[:10]  # Top 10 allocations
                ],
            }

            # Thread-safe access to memory snapshots
            with self._snapshots_lock:
                self._memory_snapshots.append(memory_info)
                # Keep only last 100 snapshots
                if len(self._memory_snapshots) > 100:
                    self._memory_snapshots = self._memory_snapshots[-100:]

            # Thread-safe access to metrics
            with self._metrics_lock:
                # Check thresholds
                if memory_mb >= self._memory_thresholds["critical_mb"]:
                    self._memory_criticals += 1
                    logger.critical(".2f")
                    self.trigger_emergency_cleanup()
                elif memory_mb >= self._memory_thresholds["warning_mb"]:
                    self._memory_warnings += 1
                    logger.warning(".2f")
                    if memory_mb >= self._memory_thresholds["cleanup_mb"]:
                        self.trigger_cleanup()

        except Exception as e:
            logger.exception(f"Error checking memory usage: {e}")

    def get_object_from_pool(
        self,
        pool_name: str,
        factory_func: callable,
        max_pool_size: int = 50,
        *args,
        **kwargs,
    ) -> Any:
        """Get an object from the pool or create a new one.

        Args:
            pool_name: Name of the object pool
            factory_func: Function to create new objects
            max_pool_size: Maximum size of the pool
            *args, **kwargs: Arguments for factory function

        Returns:
            Object from pool or newly created
        """
        with self._pools_lock:
            if pool_name not in self._object_pools:
                self._object_pools[pool_name] = []
                self._pool_sizes[pool_name] = max_pool_size

            pool = self._object_pools[pool_name]
            max_size = self._pool_sizes[pool_name]

            # Try to find an available object
            for obj in pool:
                if hasattr(obj, "_in_use") and not obj._in_use:
                    obj._in_use = True
                    logger.debug(f"Reused object from pool {pool_name}")
                    return obj

            # Create a new object if pool not full
            if len(pool) < max_size:
                try:
                    obj = factory_func(*args, **kwargs)
                    obj._in_use = True
                    obj._pool_name = pool_name
                    obj._created_time = time.time()

                    pool.append(obj)
                    with self._tracking_lock:
                        self._tracked_objects.add(obj)

                    logger.debug(f"Created new object for pool {pool_name}")
                    return obj
                except Exception as e:
                    logger.exception(
                        f"Failed to create object for pool {pool_name}: {e}"
                    )
                    return None

            # Pool is full, create temporary object
            logger.warning(f"Pool {pool_name} is full, creating temporary object")
            try:
                obj = factory_func(*args, **kwargs)
                obj._in_use = True
                obj._temporary = True
                return obj
            except Exception as e:
                logger.exception(
                    f"Failed to create temporary object for pool {pool_name}: {e}"
                )
                return None

    def return_object_to_pool(self, obj: Any):
        """Return an object to its pool.

        Args:
            obj: Object to return to pool
        """
        if not hasattr(obj, "_pool_name"):
            # Not a pooled object, just mark as not in use
            if hasattr(obj, "_in_use"):
                obj._in_use = False
            return

        pool_name = obj._pool_name
        with self._pools_lock:
            if pool_name in self._object_pools:
                obj._in_use = False
                obj._last_used = time.time()
                logger.debug(f"Returned object to pool {pool_name}")
            else:
                logger.warning(f"Pool {pool_name} not found for object return")

    def cleanup_pool(self, pool_name: str, force: bool = False):
        """Clean up a specific object pool.

        Args:
            pool_name: Name of the pool to clean up
            force: Whether to force cleanup of all objects
        """
        with self._pools_lock:
            if pool_name not in self._object_pools:
                return

            pool = self._object_pools[pool_name]
            current_time = time.time()

            if force:
                # Force cleanup all objects
                cleanup_count = 0
                for obj in pool[
                    :
                ]:  # Copy the list to avoid modification during iteration
                    if hasattr(obj, "_in_use") and not obj._in_use:
                        pool.remove(obj)
                        cleanup_count += 1

                logger.info(
                    f"Force cleaned up {cleanup_count} objects from pool {pool_name}"
                )
            else:
                # Clean up old unused objects (older than 1 hour)
                old_objects = []
                for obj in pool:
                    if (
                        hasattr(obj, "_in_use")
                        and not obj._in_use
                        and hasattr(obj, "_last_used")
                        and current_time - obj._last_used > 3600
                    ):  # 1 hour
                        old_objects.append(obj)

                for obj in old_objects:
                    pool.remove(obj)

                if old_objects:
                    logger.info(
                        f"Cleaned up {len(old_objects)} old objects from pool {pool_name}"
                    )

            self._pool_cleanup_times[pool_name] = current_time

    def cleanup_all_pools(self, force: bool = False):
        """Clean up all object pools.

        Args:
            force: Whether to force cleanup of all objects
        """
        for pool_name in list(self._object_pools.keys()):
            self.cleanup_pool(pool_name, force)

        self._cleanup_count += 1
        logger.info("Completed cleanup of all object pools")

    def trigger_cleanup(self):
        """Trigger a cleanup operation."""
        logger.info("Triggering memory cleanup")

        # Clean up object pools
        self.cleanup_all_pools()

        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Garbage collection collected {collected} objects")

        # Clean up weak references
        self._cleanup_weak_refs()

        # Call cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.exception(f"Error in cleanup callback: {e}")

    def trigger_emergency_cleanup(self):
        """Trigger emergency cleanup when memory is critical."""
        logger.critical("Triggering emergency memory cleanup")

        # Force cleanup of all pools
        self.cleanup_all_pools(force=True)

        # Aggressive garbage collection
        for _ in range(3):
            collected = gc.collect()
            logger.info(f"Emergency GC pass collected {collected} objects")

        # Clear any large caches if available
        self._clear_large_caches()

        # Final cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.exception(f"Emergency cleanup callback failed: {e}")

    def _cleanup_weak_refs(self):
        """Clean up dead weak references."""
        # Remove dead references
        dead_refs = []
        for ref_name, weak_ref in self._object_refs.items():
            if weak_ref() is None:
                dead_refs.append(ref_name)

        for ref_name in dead_refs:
            del self._object_refs[ref_name]

        if dead_refs:
            logger.debug(f"Cleaned up {len(dead_refs)} dead weak references")

    def _clear_large_caches(self):
        """Clear any large caches that may exist."""
        # This would be extended to clear specific caches
        # For now, just trigger general cleanup
        pass

    def add_cleanup_callback(self, callback: callable):
        """Add a cleanup callback function.

        Args:
            callback: Function to call during cleanup
        """
        with self._callbacks_lock:
            self._cleanup_callbacks.append(callback)

    def integrate_cache_maintenance(self, cache_instance):
        """Integrate cache maintenance with memory manager.

        Args:
            cache_instance: Cache instance with perform_maintenance method
        """

        async def cache_maintenance_callback():
            """Async callback to perform cache maintenance."""
            try:
                if hasattr(cache_instance, "perform_maintenance"):
                    result = await cache_instance.perform_maintenance()
                    if result.get("maintenance_performed", False):
                        logger.info(f"Cache maintenance completed: {result}")
            except Exception as e:
                logger.error(f"Cache maintenance failed: {str(e)}")

        # Add cache maintenance to cleanup callbacks
        with self._callbacks_lock:
            self._cleanup_callbacks.append(cache_maintenance_callback)

        logger.info("Cache maintenance integrated with memory manager")

    def track_object(self, obj: Any, name: str):
        """Track an object with a weak reference.

        Args:
            obj: Object to track
            name: Name for the object
        """
        with self._tracking_lock:
            self._object_refs[name] = ref(
                obj, lambda ref: self._on_object_deleted(name)
            )
            self._tracked_objects.add(obj)

    def _on_object_deleted(self, name: str):
        """Callback when a tracked object is deleted."""
        with self._tracking_lock:
            if name in self._object_refs:
                del self._object_refs[name]
            logger.debug(f"Tracked object {name} was deleted")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "current_memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "num_snapshots": len(self._memory_snapshots),
                "pool_stats": {
                    pool_name: {
                        "size": len(objects),
                        "max_size": self._pool_sizes.get(pool_name, 0),
                    }
                    for pool_name, objects in self._object_pools.items()
                },
                "tracked_objects": len(self._tracked_objects),
                "cleanup_count": self._cleanup_count,
                "memory_warnings": self._memory_warnings,
                "memory_criticals": self._memory_criticals,
                "thresholds": self._memory_thresholds.copy(),
            }
        except Exception as e:
            logger.exception(f"Error getting memory stats: {e}")
            return {"error": str(e)}

    def get_memory_snapshot(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent memory snapshots."""
        return self._memory_snapshots[-limit:] if self._memory_snapshots else []

    def set_memory_thresholds(
        self,
        warning_mb: float = None,
        critical_mb: float = None,
        cleanup_mb: float = None,
    ):
        """Set memory usage thresholds.

        Args:
            warning_mb: Warning threshold in MB
            critical_mb: Critical threshold in MB
            cleanup_mb: Cleanup trigger threshold in MB
        """
        if warning_mb is not None:
            self._memory_thresholds["warning_mb"] = warning_mb
        if critical_mb is not None:
            self._memory_thresholds["critical_mb"] = critical_mb
        if cleanup_mb is not None:
            self._memory_thresholds["cleanup_mb"] = cleanup_mb

        logger.info(f"Updated memory thresholds: {self._memory_thresholds}")

    async def initialize(self) -> None:
        """Initialize the memory manager asynchronously."""
        pass

    def shutdown(self):
        """Shutdown the memory manager."""
        logger.info("Shutting down MemoryManager")

        self._stop_monitoring = True

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        # Final cleanup
        self.trigger_cleanup()

        # Stop tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        logger.info("MemoryManager shutdown complete")


# Global memory manager instance
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
