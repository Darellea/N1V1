# Performance Optimization Guide for N1V1 Framework

## Overview

The N1V1 framework includes a comprehensive performance profiling and optimization system designed to ensure reliable operation under institutional workloads. This guide covers how to use the built-in profiling tools, interpret results, and implement performance optimizations.

## Key Features

- **Multi-level Profiling**: Function-level timing, memory usage tracking, I/O monitoring, and garbage collection analysis
- **Advanced Tools Integration**: Support for py-spy, scalene, memory_profiler, and flamegraph generation
- **Automated Regression Detection**: Continuous monitoring with configurable thresholds
- **Real-time Performance Monitoring**: Live metrics collection and anomaly detection
- **CI/CD Integration**: Automated profiling during testing and deployment

## Quick Start

### 1. Enable Profiling

Profiling is enabled by default in development mode. Check your `config.json`:

```json
{
  "profiling": {
    "enabled": true,
    "mode": "development",
    "sampling_interval": 0.01,
    "output_dir": "performance_reports"
  }
}
```

### 2. Basic Function Profiling

```python
from core.performance_profiler import profile_function, get_profiler

# Decorator-based profiling
@profile_function("my_function")
def my_function():
    # Your code here
    pass

# Context manager profiling
profiler = get_profiler()
with profiler.profile_function("another_function"):
    # Your code here
    pass
```

### 3. Generate Performance Report

```python
from core.performance_profiler import get_profiler

profiler = get_profiler()
report = profiler.generate_report()
print(json.dumps(report, indent=2))
```

## Profiling Tools

### Built-in Profiler

The framework includes a comprehensive built-in profiler:

```python
from core.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler(
    sampling_interval=0.01,  # 10ms sampling
    max_history=1000         # Keep 1000 measurements
)

# Start profiling session
session_id = profiler.start_profiling("my_session")

# Your code to profile
# ...

# Generate report
report = profiler.generate_report()
hotspots = profiler.get_hotspots(top_n=10)
```

### Advanced Profiling Tools

#### py-spy (Sampling Profiler)

```python
from core.performance_profiler import get_advanced_profiler

profiler = get_advanced_profiler()
process_id = profiler.start_pyspy_profiling(pid=os.getpid())

# py-spy will run for 60 seconds and generate a flamegraph
# Output: performance_reports/pyspy_<pid>_<timestamp>.svg
```

#### scalene (CPU and Memory Profiler)

```python
profiler = get_advanced_profiler()
process_id = profiler.start_scalene_profiling(script_path="main.py")

# Generates HTML report with CPU and memory profiling
# Output: performance_reports/scalene_<timestamp>/profile.html
```

#### memory_profiler (Line-by-line Memory Analysis)

```python
@profiler.profile_with_memory_profiler
def memory_intensive_function():
    data = [i for i in range(100000)]
    return sum(data)

result = memory_intensive_function()
# Output: performance_reports/memory_profile_<timestamp>.txt
```

### Flamegraph Generation

```python
from core.performance_profiler import get_profiler, get_advanced_profiler

# Get profiling data
profiler = get_profiler()
report = profiler.generate_report()

# Generate flamegraph
advanced_profiler = get_advanced_profiler()
flamegraph_path = advanced_profiler.generate_flamegraph(
    report,
    output_file="my_flamegraph.svg"
)
```

## Regression Detection

### Setting Up Baselines

```python
from core.performance_profiler import get_regression_detector

detector = get_regression_detector()

# Update baselines with current performance
metrics = {
    'execution_time': 0.05,
    'memory_usage': 1024,
    'cpu_percent': 5.0
}
detector.update_baseline("my_function", metrics)
```

### Automated Regression Detection

```python
# Check for regressions
current_metrics = {
    'execution_time': 0.07,  # 40% increase
    'memory_usage': 1024,
    'cpu_percent': 5.0
}

result = detector.detect_regression("my_function", current_metrics)

if result['regression_detected']:
    print("Performance regression detected!")
    for issue in result['issues']:
        print(f"- {issue['type']}: {issue['increase_percent']*100:.1f}% increase")
```

### Configuring Thresholds

Thresholds are configured in `config.json`:

```json
{
  "profiling": {
    "regression_detection": {
      "thresholds": {
        "latency_increase": 0.20,    // 20% latency increase threshold
        "memory_increase": 0.30,     // 30% memory increase threshold
        "cpu_increase": 0.25         // 25% CPU increase threshold
      }
    }
  }
}
```

## Interpreting Results

### Performance Report Structure

```json
{
  "summary": {
    "total_functions": 25,
    "total_measurements": 1250,
    "time_range": {
      "start": 1638360000.0,
      "end": 1638360300.0
    }
  },
  "functions": {
    "my_function": {
      "call_count": 50,
      "execution_time": {
        "total": 2.5,
        "mean": 0.05,
        "median": 0.048,
        "min": 0.04,
        "max": 0.08,
        "std": 0.01
      },
      "memory_usage": {
        "total": 51200,
        "mean": 1024,
        "peak": 2048
      },
      "performance_trends": {
        "trend": "improving",
        "slope": -0.001,
        "volatility": 0.012
      }
    }
  }
}
```

### Flamegraph Interpretation

Flamegraphs show function call stacks over time:

- **Width**: Time spent in function (wider = more time)
- **Height**: Stack depth (taller = deeper call stack)
- **Color**: Usually random (for visual distinction)

**Hotspots appear as wide bars at the top of the graph.**

### Memory Profiling Output

```
Line #    Mem usage    Increment   Line Contents
============================================================
     3     10.5 MiB      0.0 MiB   @profile
     4     10.5 MiB      0.0 MiB   def my_function():
     5     15.7 MiB      5.2 MiB       data = [i for i in range(100000)]
     6     14.2 MiB     -1.5 MiB       return sum(data)
```

## Optimization Strategies

### 1. Function-level Optimizations

```python
# Before: Inefficient list comprehension
def slow_function():
    result = [process_item(item) for item in large_dataset]
    return result

# After: Generator expression
def fast_function():
    result = (process_item(item) for item in large_dataset)
    return list(result)
```

### 2. Memory Optimization

```python
# Before: Loading all data into memory
def memory_hungry():
    data = load_all_data_from_file()
    processed = [process(d) for d in data]
    return processed

# After: Streaming processing
def memory_efficient():
    with open('data.txt') as f:
        for line in f:
            item = process(line.strip())
            yield item
```

### 3. Caching Strategies

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param):
    # Expensive calculation here
    return result
```

### 4. Async Processing

```python
import asyncio

async def async_processing():
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results
```

## CI/CD Integration

### Automated Profiling in Tests

```python
# conftest.py
import pytest
from core.performance_profiler import get_profiler

@pytest.fixture(autouse=True)
def profile_tests():
    profiler = get_profiler()
    session_id = profiler.start_profiling("test_session")
    yield
    profiler.stop_profiling()
    report = profiler.generate_report()
    # Save or analyze report
```

### Performance Regression Tests

```python
def test_performance_regression():
    from core.performance_profiler import get_regression_detector

    detector = get_regression_detector()

    # Run function under test
    start = time.time()
    result = my_function()
    execution_time = time.time() - start

    # Check for regression
    metrics = {'execution_time': execution_time}
    regression_result = detector.detect_regression("my_function", metrics)

    assert not regression_result['regression_detected'], \
        f"Performance regression: {regression_result['issues']}"
```

### GitHub Actions Example

```yaml
# .github/workflows/performance.yml
name: Performance Tests
on: [push, pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run performance tests
      run: pytest tests/test_performance_optimization.py -v
    - name: Upload profiling artifacts
      uses: actions/upload-artifact@v2
      with:
        name: performance-reports
        path: performance_reports/
```

## Monitoring and Alerting

### Prometheus Metrics

The framework exposes performance metrics to Prometheus:

```python
from core.metrics_collector import get_metrics_collector

collector = get_metrics_collector()

# Performance metrics are automatically collected
# Available at /metrics endpoint
```

### Key Metrics to Monitor

- `performance_function_execution_time_seconds`
- `performance_memory_current_bytes`
- `performance_anomalies_detected_total`
- `profiling_active_processes_count`
- `regression_functions_with_baselines_count`

### Alerting Rules

```yaml
# prometheus/alert_rules.yml
groups:
  - name: performance_alerts
    rules:
    - alert: HighFunctionLatency
      expr: performance_function_execution_time_seconds > 1.0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High function execution time detected"

    - alert: MemoryRegression
      expr: increase(performance_memory_peak_bytes[1h]) > 100000000
      for: 10m
      labels:
        severity: critical
      annotations:
        summary: "Memory usage regression detected"
```

## Troubleshooting

### Common Issues

1. **High Profiling Overhead**
   - Increase sampling interval
   - Disable memory tracking in production
   - Use selective profiling

2. **Missing Profiling Data**
   - Check if profiling is enabled in config
   - Verify tool availability (`py-spy`, `scalene`, etc.)
   - Check file permissions for output directory

3. **False Positive Regressions**
   - Increase baseline sample size
   - Adjust regression thresholds
   - Consider environmental factors (CPU load, memory pressure)

### Debugging Commands

```bash
# Check profiling status
python -c "from core.performance_profiler import get_profiler; print(get_profiler().get_profiling_status())"

# View current baselines
python -c "from core.performance_profiler import get_regression_detector; print(get_regression_detector().get_regression_report())"

# Generate flamegraph manually
python -c "from core.performance_profiler import get_advanced_profiler; get_advanced_profiler().generate_flamegraph(get_profiler().generate_report())"
```

## Best Practices

### Development

1. **Profile Early**: Start profiling during development, not just in production
2. **Set Baselines**: Establish performance baselines early in the project lifecycle
3. **Monitor Trends**: Regularly review performance trends, not just absolute values
4. **Use Realistic Data**: Profile with production-like data volumes and patterns

### Production

1. **Selective Profiling**: Only profile critical paths in production
2. **Resource Limits**: Set appropriate memory and CPU limits for profiling processes
3. **Alert Configuration**: Set up alerts for significant performance deviations
4. **Regular Reviews**: Periodically review and update performance baselines

### CI/CD

1. **Parallel Testing**: Run performance tests in parallel with functional tests
2. **Artifact Storage**: Store profiling artifacts for trend analysis
3. **Threshold Updates**: Regularly review and update performance thresholds
4. **Environment Consistency**: Ensure consistent environments for performance testing

## Advanced Topics

### Custom Profiling Decorators

```python
from functools import wraps
from core.performance_profiler import get_profiler

def profile_with_threshold(threshold_seconds=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile_function(func.__name__) as profile_ctx:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                if execution_time > threshold_seconds:
                    logger.warning(f"Function {func.__name__} exceeded threshold: {execution_time:.3f}s")

                return result
        return wrapper
    return decorator
```

### Distributed Profiling

```python
# For distributed systems
from core.performance_profiler import get_profiler

def profile_distributed_task(task_id, worker_id):
    profiler = get_profiler()

    with profiler.profile_function(f"task_{task_id}"):
        # Task execution
        result = execute_task()

        # Record distributed metrics
        profiler.record_metric(
            "distributed_task_completion_time",
            time.time(),
            {"task_id": task_id, "worker_id": worker_id}
        )

        return result
```

### Memory Leak Detection

```python
import gc
import tracemalloc

def detect_memory_leaks():
    profiler = get_profiler()

    # Take initial snapshot
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    snapshot1 = tracemalloc.take_snapshot()

    # Run your code
    # ...

    snapshot2 = tracemalloc.take_snapshot()

    # Compare snapshots
    stats = snapshot2.compare_to(snapshot1, 'lineno')

    for stat in stats[:10]:
        if stat.size_diff > 1000000:  # 1MB threshold
            logger.warning(f"Potential memory leak: {stat.traceback.format()[0]}")
```

## Support

For issues or questions about performance profiling:

1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Check the framework logs for profiling-related messages
4. Open an issue on the project repository

## Contributing

When contributing performance improvements:

1. Always include performance tests
2. Update baselines after significant changes
3. Document performance characteristics
4. Consider backward compatibility impact

---

*This guide is continuously updated. Last updated: September 2025*
