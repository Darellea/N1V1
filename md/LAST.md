---

**Prompt 1: Fix Race Condition in Signal Routing**

**Role & Context:**
- You are a **Senior Distributed Systems Engineer**. The framework's signal routing system in `router.py` has potential race conditions during high concurrency that could lead to duplicate or lost signals.

**Task / Objective:**
- Implement a message queue with idempotency checks in the signal routing system to prevent race conditions and ensure exactly-once processing.
- **Update tests/ files related to the patch**: Add comprehensive tests for race condition scenarios, duplicate signal detection, and message queue behavior in `tests/core/test_signal_router.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain backward compatibility with existing signal handlers
- Cannot change the public API of router methods
- Must handle message deduplication with configurable time windows
- Support both async and sync signal processing patterns

**Input & Expected Output Format:**
- **Input:** Current `router.py` implementation with potential race conditions
- **Output:** Modified `router.py` with message queue and idempotency checks + updated test files

**Edge Cases / Additional Conditions:**
- Handle system crashes during signal processing
- Support configurable deduplication windows (default: 5 minutes)
- Maintain signal ordering within trading pairs
- Provide metrics for queue depth and processing latency

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(30)
def test_concurrent_signal_processing(self):
    router = SignalRouter()
    signals = [TradingSignal(id=f"test_{i}") for i in range(1000)]
    
    # Process signals concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(router.route_signal, signal) for signal in signals]
        for future in as_completed(futures, timeout=10):
            future.result()  # Should not hang
```

---

**Prompt 2: Add Circuit Breaker Cooldown Enforcement**

**Role & Context:**
- You are a **Senior Reliability Engineer**. The `smart_order_executor.py` lacks proper circuit breaker cooldown period validation, risking rapid repeated failures.

**Task / Objective:**
- Add mandatory cooldown period enforcement to the circuit breaker system with exponential backoff and health checks.
- **Update tests/ files related to the patch**: Add tests for cooldown period validation, exponential backoff behavior, and circuit state transitions in `tests/core/test_circuit_breaker.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must integrate with existing circuit breaker patterns
- Cannot break existing order execution workflows
- Support configurable cooldown strategies (fixed, exponential, fibonacci)
- Provide circuit state metrics for monitoring

**Input & Expected Output Format:**
- **Input:** Current `smart_order_executor.py` circuit breaker implementation
- **Output:** Enhanced circuit breaker with cooldown enforcement and health checks + updated test files

**Edge Cases / Additional Conditions:**
- Handle partial service degradation
- Support manual circuit reset for emergencies
- Provide circuit state transitions logging
- Integrate with existing alerting system

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(10)
def test_circuit_breaker_cooldown_timeout(self):
    breaker = CircuitBreaker(cooldown_period=5)
    breaker.trip()
    
    # Should timeout quickly, not wait full cooldown
    with pytest.raises(CircuitBreakerError):
        breaker.execute_with_timeout(lambda: "test", timeout=1.0)
```

---

**Prompt 3: Fix Memory Leak in Market Data Caching**

**Role & Context:**
- You are a **Senior Performance Engineer**. The `bot_engine.py` has memory leaks in market data caching that could lead to system instability during extended runs.

**Task / Objective:**
- Implement LRU cache with size limits and memory monitoring for market data caching in the bot engine.
- **Update tests/ files related to the patch**: Add memory leak detection tests, cache eviction tests, and performance benchmarks in `tests/core/test_bot_engine_memory.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain real-time data access performance
- Cannot change data structure interfaces
- Support configurable cache sizes per data type
- Provide cache hit/miss metrics

**Input & Expected Output Format:**
- **Input:** Current market data caching implementation in `bot_engine.py`
- **Output:** Memory-safe LRU cache implementation with monitoring + updated test files

**Edge Cases / Additional Conditions:**
- Handle cache invalidation on market close
- Support emergency cache clearing
- Monitor memory usage with configurable thresholds
- Integrate with existing memory manager

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(15)
def test_cache_performance_with_timeout(self):
    cache = LRUCache(maxsize=10000)
    
    # Large batch operations with timeout
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(10000):
            future = executor.submit(cache.set, f"key_{i}", f"value_{i}")
            futures.append(future)
        
        # Wait with timeout to prevent hangs
        for future in as_completed(futures, timeout=10):
            future.result()
```

---

**Prompt 4: Implement Async Metrics Collection**

**Role & Context:**
- You are a **Senior Async Systems Engineer**. The `performance_monitor.py` uses blocking I/O for metrics collection, causing performance bottlenecks.

**Task / Objective:**
- Convert blocking metrics collection to async patterns using asyncio and proper async libraries.
- **Update tests/ files related to the patch**: Add async metric collection tests, concurrency tests, and performance comparison tests in `tests/core/test_async_metrics.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain all existing metric types and aggregations
- Cannot change metric collection API signatures
- Support both real-time and batched metric processing
- Provide performance benchmarks for comparison

**Input & Expected Output Format:**
- **Input:** Current blocking metrics collection in `performance_monitor.py`
- **Output:** Async metrics collection with non-blocking I/O + updated test files

**Edge Cases / Additional Conditions:**
- Handle metric collection during system stress
- Support graceful degradation when async operations fail
- Maintain metric accuracy during concurrent updates
- Provide async context manager support

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_async_metrics_timeout_protection(self):
    monitor = AsyncPerformanceMonitor()
    
    # Create hanging collector
    async def hanging_collector():
        await asyncio.sleep(60)  # Would hang without timeout
        return {"metric": 1}
    
    monitor.add_collector(hanging_collector)
    
    # Should timeout, not hang
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(monitor.collect_metrics(), timeout=5.0)
```

---

**Prompt 5: Add Thread-Safe Circuit Breaker State**

**Role & Context:**
- You are a **Senior Concurrency Specialist**. The `circuit_breaker.py` uses global state without proper thread safety, risking race conditions in multi-threaded environments.

**Task / Objective:**
- Implement atomic operations and proper locking for circuit breaker state management.
- **Update tests/ files related to the patch**: Add thread safety tests, race condition detection tests, and concurrent access tests in `tests/core/test_thread_safe_circuit_breaker.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain existing circuit breaker logic and thresholds
- Cannot introduce significant performance overhead
- Support both thread-local and shared circuit breakers
- Provide deadlock prevention mechanisms

**Input & Expected Output Format:**
- **Input:** Current non-thread-safe circuit breaker implementation
- **Output:** Thread-safe circuit breaker with atomic state transitions + updated test files

**Edge Cases / Additional Conditions:**
- Handle thread cancellation during state transitions
- Support reentrant locking patterns
- Provide timeout mechanisms for lock acquisition
- Integrate with existing monitoring for lock contention

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(10)
def test_lock_timeout_prevention(self):
    breaker = ThreadSafeCircuitBreaker(lock_timeout=1.0)
    
    # Acquire lock in one thread
    with breaker._state_lock:
        # Another thread should timeout, not deadlock
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(breaker.trip)
            with pytest.raises(TimeoutError):
                future.result(timeout=2.0)
```

---

**Prompt 6: Add Slippage Models to Position Sizing**

**Role & Context:**
- You are a **Senior Quantitative Analyst**. The `risk_manager.py` position sizing lacks slippage consideration, leading to inaccurate risk calculations.

**Task / Objective:**
- Implement dynamic slippage models based on market liquidity, volatility, and order size for accurate position sizing.
- **Update tests/ files related to the patch**: Add slippage model validation tests, liquidity scenario tests, and accuracy benchmarks in `tests/risk/test_slippage_models.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must integrate with existing position sizing algorithms
- Cannot change risk management API contracts
- Support multiple slippage models (constant, linear, square root)
- Provide slippage estimation accuracy metrics

**Input & Expected Output Format:**
- **Input:** Current position sizing implementation without slippage
- **Output:** Enhanced position sizing with dynamic slippage models + updated test files

**Edge Cases / Additional Conditions:**
- Handle illiquid market conditions
- Support custom slippage curves per trading pair
- Provide slippage impact analysis for large orders
- Integrate with market data for real-time liquidity assessment

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(15)
def test_slippage_calculation_performance(self):
    risk_mgr = RiskManager()
    
    # Test with large number of calculations
    start_time = time.time()
    for i in range(10000):
        size = risk_mgr.calculate_position_size_with_slippage(
            10000, 0.02, market_data=generate_market_data()
        )
        assert size > 0
    
    # Should complete within timeout
    assert time.time() - start_time < 10.0
```

---

**Prompt 7: Implement Kalman Filtering for Anomaly Detection**

**Role & Context:**
- You are a **Senior ML Engineer**. The `anomaly_detector.py` has high false positives in volatility spikes due to simplistic detection algorithms.

**Task / Objective:**
- Implement Kalman filtering and adaptive thresholding for more accurate anomaly detection in market data.
- **Update tests/ files related to the patch**: Add Kalman filter accuracy tests, false positive reduction tests, and regime change detection tests in `tests/ml/test_kalman_anomaly_detection.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain real-time detection performance
- Cannot significantly increase computational complexity
- Support online learning of market regimes
- Provide detection confidence scores

**Input & Expected Output Format:**
- **Input:** Current anomaly detection with high false positive rate
- **Output:** Kalman filter-based detection with adaptive thresholds + updated test files

**Edge Cases / Additional Conditions:**
- Handle regime changes in market behavior
- Support manual override for known market events
- Provide anomaly explanation and feature importance
- Integrate with existing alerting system

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(20)
def test_kalman_filter_real_time_performance(self):
    detector = KalmanAnomalyDetector()
    
    # Stream large dataset with timeout protection
    data_stream = generate_realtime_data_stream(count=10000)
    
    start_time = time.time()
    anomaly_count = 0
    for data_point in data_stream:
        if time.time() - start_time > 15:  # Prevent infinite loops
            break
        if detector.detect_anomaly(data_point):
            anomaly_count += 1
    
    assert anomaly_count < 100  # Should have reasonable false positive rate
```

---

**Prompt 8: Add Partial Fill Reconciliation**

**Role & Context:**
- You are a **Senior Exchange Integration Specialist**. The `order_manager.py` lacks proper validation and reconciliation for partial order fills.

**Task / Objective:**
- Implement comprehensive partial fill reconciliation with retry logic and fill validation.
- **Update tests/ files related to the patch**: Add partial fill scenario tests, reconciliation logic tests, and exchange compatibility tests in `tests/core/test_partial_fill_reconciliation.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must handle all major exchange fill patterns
- Cannot change order execution workflow interfaces
- Support configurable fill timeout and retry strategies
- Provide fill reconciliation audit trails

**Input & Expected Output Format:**
- **Input:** Current order manager without partial fill handling
- **Output:** Enhanced order manager with partial fill reconciliation + updated test files

**Edge Cases / Additional Conditions:**
- Handle exchange discrepancies in fill reporting
- Support manual intervention for stuck orders
- Provide fill reconciliation metrics and alerts
- Integrate with existing position management

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_partial_fill_timeout_handling(self):
    order_mgr = OrderManager()
    
    # Mock exchange that hangs on fill reporting
    async def hanging_fill_report():
        await asyncio.sleep(60)  # Would hang without timeout
        return Fill(quantity=0.5, order_id="test")
    
    # Should timeout gracefully
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            order_mgr.process_fill_report(hanging_fill_report), 
            timeout=5.0
        )
```

---

**Prompt 9: Implement Gradual Policy Transitions**

**Role & Context:**
- You are a **Senior Systems Architect**. The `adaptive_policy.py` has abrupt risk policy changes that could cause trading disruptions.

**Task / Objective:**
- Implement gradual policy transitions with smooth interpolation between policy states.
- **Update tests/ files related to the patch**: Add policy transition tests, interpolation accuracy tests, and emergency override tests in `tests/core/test_gradual_policy_transitions.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain policy consistency during transitions
- Cannot introduce significant latency in policy application
- Support both immediate and gradual transition modes
- Provide transition progress monitoring

**Input & Expected Output Format:**
- **Input:** Current policy implementation with abrupt changes
- **Output:** Enhanced policy system with gradual transitions + updated test files

**Edge Cases / Additional Conditions:**
- Handle emergency policy overrides
- Support rollback of failed policy transitions
- Provide transition validation and safety checks
- Integrate with existing risk controls

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(10)
def test_policy_transition_timeout_safety(self):
    policy_mgr = AdaptivePolicyManager()
    
    # Create transition that would normally take too long
    long_transition = PolicyTransition(
        from_policy=RiskPolicy(max_position=1000),
        to_policy=RiskPolicy(max_position=2000),
        duration=300  # 5 minutes
    )
    
    # Should allow interruption and timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(policy_mgr.execute_transition, long_transition)
        time.sleep(1)  # Let it start
        future.cancel()  # Should handle cancellation gracefully
        
        with pytest.raises((CancelledError, TimeoutError)):
            future.result(timeout=2.0)
```

---

**Prompt 10: Add Retry with Exponential Backoff for API Calls**

**Role & Context:**
- You are a **Senior API Integration Engineer**. The `data_fetcher.py` lacks proper handling for API rate limits and temporary failures.

**Task / Objective:**
- Implement retry mechanism with exponential backoff, jitter, and circuit breaking for all external API calls.
- **Update tests/ files related to the patch**: Add retry behavior tests, rate limit handling tests, and circuit breaker integration tests in `tests/integration/test_api_retry_mechanisms.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must respect API rate limits and terms of service
- Cannot change data fetching interface signatures
- Support configurable retry strategies per API endpoint
- Provide detailed retry metrics and logging

**Input & Expected Output Format:**
- **Input:** Current API calling implementation without proper retry logic
- **Output:** Robust API client with exponential backoff and rate limit handling + updated test files

**Edge Cases / Additional Conditions:**
- Handle permanent vs temporary failures differently
- Support request deduplication for idempotent operations
- Provide graceful degradation when APIs are unavailable
- Integrate with existing circuit breaker system

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(15)
def test_retry_timeout_prevention(self):
    fetcher = DataFetcher(
        retry_strategy=ExponentialBackoff(max_retries=10, max_wait=60)
    )
    
    # Mock API that always fails but would cause long retry cycle
    mock_session = Mock()
    mock_session.get.side_effect = RateLimitError("Too many requests")
    
    # Should timeout before completing all retries
    start_time = time.time()
    with pytest.raises((TimeoutError, MaxRetriesExceeded)):
        with timeout(5.0):  # Custom timeout context
            fetcher.fetch_data_with_retry("test_endpoint", session=mock_session)
    
    assert time.time() - start_time < 6.0  # Should respect timeout
```

---

**Prompt 11: Fix Unbounded Memory Growth in Historical Data Loading**

**Role & Context:**
- You are a **Senior Data Engineer**. The `historical_loader.py` suffers from unbounded memory growth when processing large datasets, risking system crashes.

**Task / Objective:**
- Implement streaming data processing with chunked loading and memory-efficient data structures for historical data operations.
- **Update tests/ files related to the patch**: Add memory usage monitoring tests, chunked processing validation, and large dataset handling tests in `tests/data/test_memory_efficient_loading.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain data integrity and ordering
- Cannot change the historical data API interface
- Support configurable chunk sizes based on available memory
- Provide memory usage monitoring during loading

**Input & Expected Output Format:**
- **Input:** Current memory-intensive historical data loading
- **Output:** Memory-efficient streaming data loader with chunked processing + updated test files

**Edge Cases / Additional Conditions:**
- Handle corrupted data chunks gracefully
- Support resume capability for interrupted loads
- Provide progress tracking for large datasets
- Integrate with existing memory manager

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(30)
def test_large_dataset_loading_with_timeout(self):
    loader = HistoricalDataLoader(max_memory_mb=100)
    # Generate dataset that would normally cause memory issues
    large_dataset = generate_large_dataset(size_gb=10)
    
    # Process with timeout protection
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            lambda: list(loader.load_chunked(large_dataset, chunk_size=1000))
        )
        chunks = future.result(timeout=25)  # Should complete within timeout
        assert len(chunks) > 0
```

---

**Prompt 12: Add Version Locking for Dataset Updates**

**Role & Context:**
- You are a **Senior Database Architect**. The `dataset_versioning.py` has race conditions during concurrent version updates that could corrupt datasets.

**Task / Objective:**
- Implement distributed locking mechanism for dataset version operations to prevent concurrent modification conflicts.
- **Update tests/ files related to the patch**: Add concurrent update tests, lock timeout tests, and deadlock detection tests in `tests/data/test_dataset_versioning_locks.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must support multiple concurrent readers
- Cannot introduce significant latency for read operations
- Support lock timeouts and deadlock detection
- Provide lock acquisition metrics

**Input & Expected Output Format:**
- **Input:** Current versioning system without proper locking
- **Output:** Thread-safe versioning with distributed locks + updated test files

**Edge Cases / Additional Conditions:**
- Handle lock timeouts gracefully
- Support lock priority for critical operations
- Provide lock debugging and monitoring
- Integrate with existing distributed coordination

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(15)
def test_lock_timeout_prevention(self):
    version_mgr = DatasetVersionManager(lock_timeout=5)
    
    # Acquire lock and try to acquire again (should timeout)
    with version_mgr.acquire_lock("test_dataset"):
        # Second attempt should timeout, not hang
        start_time = time.time()
        with pytest.raises(LockTimeoutError):
            with version_mgr.acquire_lock("test_dataset", timeout=2):
                pass
        assert time.time() - start_time < 3.0  # Should respect timeout
```

---

**Prompt 13: Add Real-Time Model Performance Tracking**

**Role & Context:**
- You are a **Senior MLOps Engineer**. The `model_monitor.py` has delays in model drift detection, risking trading with stale models.

**Task / Objective:**
- Implement real-time performance tracking with streaming metrics and immediate drift alerts for ML models.
- **Update tests/ files related to the patch**: Add real-time drift detection tests, streaming metrics accuracy tests, and alert timing tests in `tests/ml/test_realtime_model_monitoring.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain model inference performance
- Cannot change model prediction interfaces
- Support multiple drift detection algorithms
- Provide real-time performance dashboards

**Input & Expected Output Format:**
- **Input:** Current batch-based model monitoring
- **Output:** Real-time streaming model performance monitoring + updated test files

**Edge Cases / Additional Conditions:**
- Handle concept drift vs data drift differently
- Support model performance benchmarking
- Provide automated model retraining triggers
- Integrate with existing alerting system

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(20)
def test_streaming_metrics_performance(self):
    monitor = RealTimeModelMonitor()
    
    # Process high-frequency predictions with timeout
    start_time = time.time()
    prediction_count = 0
    
    while time.time() - start_time < 15:  # Prevent infinite loops
        monitor.record_prediction(
            features=[random.random() for _ in range(10)],
            prediction=random.random(),
            actual=random.random()
        )
        prediction_count += 1
        if prediction_count > 10000:  # Safety limit
            break
    
    # Should process efficiently without hanging
    assert prediction_count > 1000
    assert monitor.get_metrics() is not None
```

---

**Prompt 14: Ensure Deterministic Optimization Results**

**Role & Context:**
- You are a **Senior Research Engineer**. The `genetic_optimizer.py` produces non-deterministic results due to random seed issues, making backtests irreproducible.

**Task / Objective:**
- Implement strict random seed control and deterministic execution for all optimization algorithms.
- **Update tests/ files related to the patch**: Add reproducibility tests, seed isolation tests, and deterministic benchmark tests in `tests/optimization/test_deterministic_optimization.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain optimization algorithm effectiveness
- Cannot change optimization result quality
- Support both reproducible and exploratory modes
- Provide seed management across distributed workers

**Input & Expected Output Format:**
- **Input:** Current non-deterministic genetic optimizer
- **Output:** Deterministic optimizer with controlled randomness + updated test files

**Edge Cases / Additional Conditions:**
- Handle parallel execution with seed isolation
- Support random state checkpointing and restoration
- Provide reproducibility validation tests
- Integrate with existing experiment tracking

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(60)
def test_optimization_timeout_safety(self):
    optimizer = GeneticOptimizer(
        population_size=1000,
        generations=1000  # Large optimization that could hang
    )
    
    # Should complete within timeout or fail gracefully
    try:
        result = optimizer.optimize(
            objective_function=complex_calculation,
            timeout=45  # Internal timeout
        )
        assert result is not None
    except TimeoutError:
        # Should raise timeout rather than hang indefinitely
        pass
```

---

**Prompt 15: Vectorize Feature Calculations**

**Role & Context:**
- You are a **Senior Performance Optimization Engineer**. The `features.py` uses inefficient loop-based calculations that bottleneck feature engineering.

**Task / Objective:**
- Vectorize all feature calculations using NumPy/Pandas operations and eliminate Python loops for performance gains.
- **Update tests/ files related to the patch**: Add performance benchmark tests, numerical accuracy tests, and edge case handling tests in `tests/features/test_vectorized_operations.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain exact numerical accuracy
- Cannot change feature definitions or interfaces
- Support both real-time and batch processing modes
- Provide performance benchmarks

**Input & Expected Output Format:**
- **Input:** Current loop-based feature calculations
- **Output:** Vectorized feature engineering with significant performance improvement + updated test files

**Edge Cases / Additional Conditions:**
- Handle NaN values and edge cases in vectorized operations
- Support incremental updates for streaming data
- Provide memory usage optimization
- Integrate with existing data pipelines

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(30)
def test_vectorized_large_dataset_performance(self):
    calculator = VectorizedFeatureCalculator()
    
    # Large dataset that would timeout with loops
    large_prices = np.random.random(1000000) * 1000  # 1M data points
    
    start_time = time.time()
    features = calculator.calculate_all_features(large_prices)
    processing_time = time.time() - start_time
    
    # Should complete within timeout and be faster than threshold
    assert processing_time < 10.0
    assert len(features) == len(large_prices)
```

---

**Prompt 16: Implement Comprehensive Order Validation Pipeline**

**Role & Context:**
- You are a **Senior Trading Systems Architect**. The framework lacks comprehensive order validation, risking invalid orders reaching exchanges.

**Task / Objective:**
- Implement multi-stage order validation pipeline with pre-trade checks, risk validation, and exchange compatibility verification.
- **Update tests/ files related to the patch**: Add validation stage tests, exchange-specific rule tests, and latency measurement tests in `tests/orders/test_comprehensive_validation.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain low latency for order execution
- Cannot bypass existing risk controls
- Support configurable validation rules per exchange
- Provide detailed validation failure reasons

**Input & Expected Output Format:**
- **Input:** Current minimal order validation
- **Output:** Comprehensive multi-stage order validation pipeline + updated test files

**Edge Cases / Additional Conditions:**
- Handle market hour restrictions
- Support exchange-specific order constraints
- Provide validation rule management interface
- Integrate with real-time market data

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(10)
def test_validation_pipeline_performance(self):
    validator = OrderValidationPipeline()
    
    # Test with many orders to ensure no performance degradation
    orders = [generate_valid_order() for _ in range(1000)]
    
    start_time = time.time()
    results = []
    for order in orders:
        try:
            result = validator.validate(order)
            results.append(result)
        except ValidationError:
            results.append(False)
    
    processing_time = time.time() - start_time
    # Should validate 1000 orders within timeout
    assert processing_time < 5.0
    assert len(results) == 1000
```

---

**Prompt 17: Add Distributed Circuit Breaker Coordination**

**Role & Context:**
- You are a **Senior Distributed Systems Architect**. Circuit breakers operate in isolation, risking inconsistent states across distributed components.

**Task / Objective:**
- Implement distributed circuit breaker coordination with consensus and state synchronization across all framework instances.
- **Update tests/ files related to the patch**: Add distributed state sync tests, network partition tests, and consensus algorithm tests in `tests/distributed/test_circuit_breaker_coordination.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain circuit breaker responsiveness
- Cannot introduce significant coordination overhead
- Support partial failure scenarios
- Provide distributed state consistency guarantees

**Input & Expected Output Format:**
- **Input:** Current isolated circuit breaker implementation
- **Output:** Distributed circuit breaker coordination system + updated test files

**Edge Cases / Additional Conditions:**
- Handle network partitions gracefully
- Support manual circuit override in emergencies
- Provide circuit state visualization
- Integrate with service discovery

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(20)
def test_distributed_consensus_timeout(self):
    nodes = [DistributedCircuitBreaker(node_id=i) for i in range(5)]
    
    # Simulate network partition - some nodes unreachable
    with patch.object(nodes[2], 'broadcast_state', side_effect=NetworkError):
        nodes[0].trip()
        
        # Should reach consensus within timeout despite partition
        wait_until(lambda: all(n.state == "open" for n in [nodes[0], nodes[1], nodes[3], nodes[4]]),
                  timeout=10.0)
        
        assert nodes[2].state == "closed"  # Partitioned node remains unchanged
```

---

**Prompt 18: Enforce Strict Memory Limits**

**Role & Context:**
- You are a **Senior Systems Reliability Engineer**. The framework lacks strict memory limits, risking out-of-memory crashes during high load.

**Task / Objective:**
- Implement hard memory limits with proactive monitoring and graceful degradation when limits are approached.
- **Update tests/ files related to the patch**: Add memory limit enforcement tests, graceful degradation tests, and emergency cleanup tests in `tests/system/test_memory_limits.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain system functionality under memory pressure
- Cannot cause uncontrolled crashes
- Support configurable limits per component
- Provide memory usage forecasting

**Input & Expected Output Format:**
- **Input:** Current memory usage without hard limits
- **Output:** Memory-limited system with graceful degradation + updated test files

**Edge Cases / Additional Conditions:**
- Handle memory pressure during critical operations
- Support emergency memory cleanup procedures
- Provide memory usage alerts and trends
- Integrate with existing monitoring

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(15)
def test_memory_cleanup_timeout(self):
    memory_manager = MemoryManager(limit_mb=50)
    
    # Allocate memory then trigger cleanup
    large_objects = [bytearray(10 * 1024 * 1024) for _ in range(10)]  # 100MB total
    
    # Cleanup should complete within timeout
    start_time = time.time()
    memory_manager.force_cleanup(timeout=10.0)
    cleanup_time = time.time() - start_time
    
    assert cleanup_time < 11.0  # Should respect timeout
    assert memory_manager.get_memory_usage() < 50 * 1024 * 1024
```

---

**Prompt 19: Implement Comprehensive Error Recovery**

**Role & Context:**
- You are a **Senior Resilience Engineer**. The framework lacks comprehensive error recovery procedures, risking extended downtime after failures.

**Task / Objective:**
- Implement systematic error recovery with state restoration, cleanup, and resumption capabilities for all major components.
- **Update tests/ files related to the patch**: Add recovery procedure tests, state restoration tests, and failure scenario tests in `tests/system/test_error_recovery.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain data consistency during recovery
- Cannot lose critical state information
- Support automated and manual recovery modes
- Provide recovery time objectives (RTO) tracking

**Input & Expected Output Format:**
- **Input:** Current limited error handling
- **Output:** Comprehensive error recovery system + updated test files

**Edge Cases / Additional Conditions:**
- Handle partial recovery scenarios
- Support recovery validation and testing
- Provide recovery procedure documentation
- Integrate with existing alerting

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(30)
def test_recovery_timeout_safety(self):
    recovery_manager = RecoveryManager()
    
    # Simulate recovery process that could hang
    async def hanging_recovery():
        await asyncio.sleep(60)  # Would hang without timeout
        return True
    
    # Should timeout gracefully
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            recovery_manager.execute_recovery_procedure(hanging_recovery),
            timeout=5.0
        )
    
    # System should be in safe state after timeout
    assert recovery_manager.state == "safe"
```

---

**Prompt 20: Add Idempotent Message Processing**

**Role & Context:**
- You are a **Senior Message Systems Engineer**. The framework lacks idempotent message processing, risking duplicate processing of signals and orders.

**Task / Objective:**
- Implement idempotent message processing with deduplication, exactly-once semantics, and idempotency keys for all critical messages.
- **Update tests/ files related to the patch**: Add duplicate detection tests, idempotency guarantee tests, and message replay tests in `tests/messaging/test_idempotent_processing.py`.
- **and all the updated tests must passed. use timeout to prevent hangs**

**Constraints / Rules:**
- Must maintain message processing performance
- Cannot lose legitimate messages
- Support configurable deduplication windows
- Provide duplicate detection metrics

**Input & Expected Output Format:**
- **Input:** Current non-idempotent message processing
- **Output:** Idempotent message processing system + updated test files

**Edge Cases / Additional Conditions:**
- Handle message replay attacks
- Support idempotency key generation and validation
- Provide message processing idempotency guarantees
- Integrate with existing message queues

**Examples / Snippets:**
```python
# Test case to add with timeout:
@pytest.mark.timeout(15)
def test_duplicate_detection_performance(self):
    processor = IdempotentMessageProcessor()
    
    # Process many messages with duplicates to test performance
    messages = []
    for i in range(1000):
        msg = TradingSignal(id=f"msg_{i % 100}")  # Create duplicates
        messages.append(msg)
    
    start_time = time.time()
    results = []
    for msg in messages:
        result = asyncio.run(processor.process(msg))
        results.append(result)
    
    processing_time = time.time() - start_time
    
    # Should process efficiently within timeout
    assert processing_time < 10.0
    # Verify correct duplicate detection
    assert sum(1 for r in results if r.executed) == 100  # Only 100 unique messages
```

---