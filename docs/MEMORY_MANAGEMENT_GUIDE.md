# Memory Management Guide for N1V1 Framework

## Overview

This guide documents the memory management patterns and best practices implemented in the N1V1 Framework, particularly focusing on the AsyncOptimizer component and memory leak prevention strategies.

## Memory Leak Prevention Architecture

### Core Components

#### 1. AsyncOptimizer Memory Monitoring
- **Automatic Memory Tracking**: Monitors memory usage on every operation
- **Threshold-Based Cleanup**: Triggers cleanup when memory thresholds are exceeded
- **Periodic Maintenance**: Regular cleanup cycles to prevent accumulation
- **Comprehensive Reporting**: Detailed memory usage statistics and recommendations

#### 2. Memory Thresholds
```python
_memory_thresholds = {
    "warning_mb": 500,      # Warn at 500MB
    "critical_mb": 1000,    # Critical cleanup at 1GB
    "cleanup_interval": 300  # Cleanup every 5 minutes
}
```

#### 3. Automatic Cleanup Triggers
- **Critical Threshold**: Immediate aggressive cleanup when memory > 1GB
- **Warning Threshold**: Logging alerts when memory > 500MB
- **Periodic Cleanup**: Light cleanup every 5 minutes
- **Operation-Based**: Memory check on every recorded operation

## Implementation Patterns

### 1. Resource Cleanup in Async Operations

#### Pattern: Proper Task Cancellation
```python
async def shutdown(self):
    """Shutdown with proper task cleanup."""
    # Cancel pending tasks
    current_task = asyncio.current_task()
    if current_task:
        all_tasks = asyncio.all_tasks()
        optimizer_tasks = [task for task in all_tasks
                         if 'AsyncOpt' in str(task.get_coro())]

        for task in optimizer_tasks:
            if not task.done() and task != current_task:
                task.cancel()

        # Wait for cancellation
        if optimizer_tasks:
            await asyncio.gather(*optimizer_tasks, return_exceptions=True)
```

#### Pattern: Reference Clearing
```python
# Clear references to help GC
self._operation_stats.clear()
self._blocking_operations.clear()
self._performance_metrics.clear()
```

### 2. Memory Monitoring Integration

#### Pattern: Operation-Based Monitoring
```python
def _record_operation(self, operation_type: str, execution_time: float):
    # Record operation stats
    # ... operation recording logic ...

    # Check memory usage and trigger cleanup if needed
    self._check_memory_usage()
```

#### Pattern: Threshold-Based Actions
```python
def _check_memory_usage(self):
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024

    if memory_mb > self._memory_thresholds["critical_mb"]:
        logger.warning(f"Critical memory usage: {memory_mb:.1f}MB")
        self._perform_memory_cleanup()
    elif memory_mb > self._memory_thresholds["warning_mb"]:
        logger.info(f"High memory usage: {memory_mb:.1f}MB")
```

### 3. Garbage Collection Management

#### Pattern: Aggressive Cleanup
```python
def _perform_memory_cleanup(self):
    # Force garbage collection
    collected = gc.collect()
    logger.info(f"Garbage collection collected {collected} objects")

    # Clear large data structures
    for op_type in list(self._operation_stats.keys()):
        if len(self._operation_stats[op_type]) > 500:
            self._operation_stats[op_type] = self._operation_stats[op_type][-250:]
```

#### Pattern: Periodic Maintenance
```python
def _perform_periodic_cleanup(self):
    # Light garbage collection
    collected = gc.collect(0)  # Only collect generation 0
    if collected > 0:
        logger.debug(f"Periodic GC collected {collected} objects")
```

## Best Practices for Future Development

### 1. Resource Management

#### Always Use Context Managers
```python
# ✅ Good: Proper cleanup
async with aiofiles.open(file_path, 'r') as f:
    content = await f.read()

# ❌ Bad: Potential resource leak
f = await aiofiles.open(file_path, 'r')
content = await f.read()
# Forgot to close f
```

#### Implement Proper Shutdown Methods
```python
class MyComponent:
    async def shutdown(self):
        """Cleanup resources properly."""
        # Cancel tasks
        # Close connections
        # Clear references
        pass

    def __del__(self):
        """Fallback cleanup."""
        try:
            if hasattr(self, 'shutdown'):
                # Note: __del__ is sync, so we can't await
                # Use sync cleanup or schedule async cleanup
                pass
        except:
            pass
```

### 2. Memory Monitoring

#### Add Memory Checks to Long-Running Operations
```python
async def long_running_operation(self):
    start_memory = psutil.Process().memory_info().rss

    # ... operation logic ...

    end_memory = psutil.Process().memory_info().rss
    memory_increase = end_memory - start_memory

    if memory_increase > 50 * 1024 * 1024:  # 50MB
        logger.warning(f"High memory increase: {memory_increase / 1024 / 1024:.1f}MB")
```

#### Use Memory Profiling in Development
```python
import tracemalloc

def profile_memory(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = await func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        logger.info(f"Memory usage: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")
        tracemalloc.stop()
        return result
    return wrapper
```

### 3. Data Structure Management

#### Limit Collection Sizes
```python
# ✅ Good: Bounded collections
if len(self._operation_stats[op_type]) > 1000:
    self._operation_stats[op_type] = self._operation_stats[op_type][-1000:]

# ❌ Bad: Unbounded growth
self._operation_stats[op_type].append(execution_time)  # Grows indefinitely
```

#### Use Weak References for Caches
```python
import weakref

class CacheWithWeakRefs:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        self._cache[key] = value
```

### 4. Async Task Management

#### Avoid Task Accumulation
```python
# ✅ Good: Track and cleanup tasks
self._active_tasks = set()

async def run_task(self, coro):
    task = asyncio.create_task(coro)
    self._active_tasks.add(task)

    try:
        return await task
    finally:
        self._active_tasks.discard(task)

# ❌ Bad: Fire and forget
async def run_task(self, coro):
    asyncio.create_task(coro)  # Task may never be cleaned up
```

#### Use Task Groups for Related Operations
```python
async def process_batch(self, items):
    async with asyncio.TaskGroup() as tg:
        for item in items:
            tg.create_task(self.process_item(item))
    # All tasks are properly awaited and cleaned up
```

## Memory Leak Detection and Testing

### 1. Stress Testing Pattern
```python
class MemoryLeakTest:
    async def run_stress_test(self):
        initial_memory = psutil.Process().memory_info().rss

        # Run operations that may cause memory leaks
        await self.perform_operations()

        final_memory = psutil.Process().memory_info().rss
        memory_increase = final_memory - initial_memory

        # Assert acceptable memory increase
        assert memory_increase < 100 * 1024 * 1024  # < 100MB
```

### 2. Memory Profiling in Tests
```python
@pytest.fixture
def memory_profiler():
    import tracemalloc
    tracemalloc.start()
    yield
    current, peak = tracemalloc.get_traced_memory()
    print(f"Memory: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")
    tracemalloc.stop()
```

### 3. Integration with CI/CD
```yaml
# .github/workflows/memory-test.yml
- name: Run Memory Leak Tests
  run: |
    python -m pytest tests/test_memory_leak_stress.py -v
    python -c "import gc; print(f'GC collections: {gc.get_count()}')"
```

## Monitoring and Alerting

### 1. Memory Metrics Collection
```python
def collect_memory_metrics():
    process = psutil.Process()
    return {
        "rss_mb": process.memory_info().rss / 1024 / 1024,
        "vms_mb": process.memory_info().vms / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "gc_collections": gc.get_count(),
        "gc_thresholds": gc.get_threshold()
    }
```

### 2. Alert Configuration
```python
MEMORY_ALERTS = {
    "critical": {
        "threshold_mb": 1000,
        "action": "immediate_cleanup",
        "notification": "slack_alert"
    },
    "warning": {
        "threshold_mb": 500,
        "action": "log_warning",
        "notification": "email_alert"
    }
}
```

## Performance Considerations

### 1. Memory Monitoring Overhead
- Memory checks are performed on every operation
- Use sampling for high-frequency operations:
```python
self._memory_check_counter = 0

def _should_check_memory(self):
    self._memory_check_counter += 1
    return self._memory_check_counter % 10 == 0  # Check every 10 operations
```

### 2. Cleanup Performance
- Aggressive cleanup may cause GC pauses
- Balance cleanup frequency with performance requirements
- Consider incremental cleanup for large data structures

### 3. Process Pool Memory
- Process pools can consume significant memory
- Monitor process memory usage separately
- Consider process recycling for long-running applications

## Troubleshooting Memory Issues

### Common Patterns and Solutions

#### 1. Thread Pool Memory Leaks
```python
# Problem: ThreadPoolExecutor not properly shutdown
# Solution: Always call shutdown() with wait=True
await self._thread_pool.shutdown(wait=True)
```

#### 2. Circular References
```python
# Problem: Objects reference each other preventing GC
# Solution: Use weak references or explicit cleanup
import weakref

class Component:
    def __init__(self, parent):
        self._parent_ref = weakref.ref(parent)
```

#### 3. Large Object Accumulation
```python
# Problem: Collections grow without bounds
# Solution: Implement size limits and cleanup policies
def add_item(self, item):
    self._items.append(item)
    if len(self._items) > self._max_size:
        # Remove oldest items or implement LRU
        self._items = self._items[-self._max_size:]
```

#### 4. Async Task Leaks
```python
# Problem: Tasks created but never awaited
# Solution: Track all tasks and ensure cleanup
self._pending_tasks = set()

def create_task(self, coro):
    task = asyncio.create_task(coro)
    self._pending_tasks.add(task)
    task.add_done_callback(self._pending_tasks.discard)
```

## Future Enhancements

### 1. Advanced Memory Profiling
- Integration with memory_profiler
- Heap analysis and object tracking
- Memory usage visualization

### 2. Distributed Memory Monitoring
- Cross-process memory coordination
- Cluster-wide memory management
- Memory-aware load balancing

### 3. Machine Learning-Based Optimization
- Predictive memory usage modeling
- Automatic threshold adjustment
- Anomaly detection for memory leaks

## Conclusion

The N1V1 Framework implements comprehensive memory management patterns that prevent memory leaks while maintaining high performance. Key principles include:

1. **Proactive Monitoring**: Continuous memory tracking with automatic cleanup
2. **Proper Resource Management**: Context managers and explicit cleanup
3. **Bounded Data Structures**: Size limits to prevent unbounded growth
4. **Comprehensive Testing**: Stress tests to verify leak prevention
5. **Performance Awareness**: Balancing monitoring overhead with system performance

Following these patterns ensures the framework remains memory-efficient and stable under production loads.
