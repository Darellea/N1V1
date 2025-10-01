Here are the updated prompts with the new "update tests/ files related to the patch" detail added to each one:

---

**Prompt 1: Fix Race Condition in Signal Routing**

**Role & Context:**
- You are a **Senior Distributed Systems Engineer**. The framework's signal routing system in `router.py` has potential race conditions during high concurrency that could lead to duplicate or lost signals.

**Task / Objective:**
- Implement a message queue with idempotency checks in the signal routing system to prevent race conditions and ensure exactly-once processing.
- **Update tests/ files related to the patch**: Add comprehensive tests for race condition scenarios, duplicate signal detection, and message queue behavior in `tests/core/test_signal_router.py`.

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
# Test case to add:
def test_duplicate_signal_detection(self):
    signal = TradingSignal(id="test_123", symbol="BTC/USDT")
    # Send same signal twice
    await router.route_signal(signal)
    await router.route_signal(signal)
    # Verify only processed once
    assert processor.call_count == 1
```

---

**Prompt 2: Add Circuit Breaker Cooldown Enforcement**

**Role & Context:**
- You are a **Senior Reliability Engineer**. The `smart_order_executor.py` lacks proper circuit breaker cooldown period validation, risking rapid repeated failures.

**Task / Objective:**
- Add mandatory cooldown period enforcement to the circuit breaker system with exponential backoff and health checks.
- **Update tests/ files related to the patch**: Add tests for cooldown period validation, exponential backoff behavior, and circuit state transitions in `tests/core/test_circuit_breaker.py`.

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
# Test case to add:
def test_circuit_breaker_cooldown_respect(self):
    breaker = CircuitBreaker(cooldown_period=60)
    breaker.trip()
    # Attempt execution during cooldown
    with pytest.raises(CircuitBreakerError):
        breaker.execute(lambda: "test")
```

---

**Prompt 3: Fix Memory Leak in Market Data Caching**

**Role & Context:**
- You are a **Senior Performance Engineer**. The `bot_engine.py` has memory leaks in market data caching that could lead to system instability during extended runs.

**Task / Objective:**
- Implement LRU cache with size limits and memory monitoring for market data caching in the bot engine.
- **Update tests/ files related to the patch**: Add memory leak detection tests, cache eviction tests, and performance benchmarks in `tests/core/test_bot_engine_memory.py`.

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
# Test case to add:
def test_cache_memory_limits(self):
    cache = LRUCache(maxsize=1000)
    # Add data beyond limit
    for i in range(2000):
        cache[f"key_{i}"] = "x" * 100
    assert len(cache) <= 1000  # Should respect size limit
```

---

**Prompt 4: Implement Async Metrics Collection**

**Role & Context:**
- You are a **Senior Async Systems Engineer**. The `performance_monitor.py` uses blocking I/O for metrics collection, causing performance bottlenecks.

**Task / Objective:**
- Convert blocking metrics collection to async patterns using asyncio and proper async libraries.
- **Update tests/ files related to the patch**: Add async metric collection tests, concurrency tests, and performance comparison tests in `tests/core/test_async_metrics.py`.

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
# Test case to add:
@pytest.mark.asyncio
async def test_async_metrics_concurrent_collection(self):
    monitor = AsyncPerformanceMonitor()
    # Collect metrics from multiple coroutines simultaneously
    tasks = [monitor.collect_metrics() for _ in range(100)]
    results = await asyncio.gather(*tasks)
    assert all(r is not None for r in results)
```

---

**Prompt 5: Add Thread-Safe Circuit Breaker State**

**Role & Context:**
- You are a **Senior Concurrency Specialist**. The `circuit_breaker.py` uses global state without proper thread safety, risking race conditions in multi-threaded environments.

**Task / Objective:**
- Implement atomic operations and proper locking for circuit breaker state management.
- **Update tests/ files related to the patch**: Add thread safety tests, race condition detection tests, and concurrent access tests in `tests/core/test_thread_safe_circuit_breaker.py`.

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
# Test case to add:
def test_circuit_breaker_thread_safety(self):
    breaker = ThreadSafeCircuitBreaker()
    def worker():
        for _ in range(1000):
            breaker.trip()
            breaker.reset()
    
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    # Should not have race conditions or corrupted state
```

---

**Prompt 6: Add Slippage Models to Position Sizing**

**Role & Context:**
- You are a **Senior Quantitative Analyst**. The `risk_manager.py` position sizing lacks slippage consideration, leading to inaccurate risk calculations.

**Task / Objective:**
- Implement dynamic slippage models based on market liquidity, volatility, and order size for accurate position sizing.
- **Update tests/ files related to the patch**: Add slippage model validation tests, liquidity scenario tests, and accuracy benchmarks in `tests/risk/test_slippage_models.py`.

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
# Test case to add:
def test_slippage_impact_on_position_sizing(self):
    risk_mgr = RiskManager()
    # Test with different market conditions
    size_without_slippage = risk_mgr.calculate_position_size(10000, 0.02)
    size_with_slippage = risk_mgr.calculate_position_size_with_slippage(
        10000, 0.02, liquidity="low"
    )
    assert size_with_slippage < size_without_slippage
```

---

**Prompt 7: Implement Kalman Filtering for Anomaly Detection**

**Role & Context:**
- You are a **Senior ML Engineer**. The `anomaly_detector.py` has high false positives in volatility spikes due to simplistic detection algorithms.

**Task / Objective:**
- Implement Kalman filtering and adaptive thresholding for more accurate anomaly detection in market data.
- **Update tests/ files related to the patch**: Add Kalman filter accuracy tests, false positive reduction tests, and regime change detection tests in `tests/ml/test_kalman_anomaly_detection.py`.

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
# Test case to add:
def test_kalman_filter_false_positive_reduction(self):
    detector = KalmanAnomalyDetector()
    normal_data = generate_normal_market_data()
    # Should not flag normal volatility as anomaly
    anomalies = detector.detect(normal_data)
    assert len(anomalies) == 0  # No false positives
```

---

**Prompt 8: Add Partial Fill Reconciliation**

**Role & Context:**
- You are a **Senior Exchange Integration Specialist**. The `order_manager.py` lacks proper validation and reconciliation for partial order fills.

**Task / Objective:**
- Implement comprehensive partial fill reconciliation with retry logic and fill validation.
- **Update tests/ files related to the patch**: Add partial fill scenario tests, reconciliation logic tests, and exchange compatibility tests in `tests/core/test_partial_fill_reconciliation.py`.

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
# Test case to add:
def test_partial_fill_reconciliation(self):
    order_mgr = OrderManager()
    partial_fill = Fill(quantity=0.5, order_id="test_123")
    # Process partial fill and verify reconciliation
    await order_mgr.process_fill(partial_fill)
    assert order_mgr.get_pending_quantity("test_123") == 0.5
```

---

**Prompt 9: Implement Gradual Policy Transitions**

**Role & Context:**
- You are a **Senior Systems Architect**. The `adaptive_policy.py` has abrupt risk policy changes that could cause trading disruptions.

**Task / Objective:**
- Implement gradual policy transitions with smooth interpolation between policy states.
- **Update tests/ files related to the patch**: Add policy transition tests, interpolation accuracy tests, and emergency override tests in `tests/core/test_gradual_policy_transitions.py`.

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
# Test case to add:
def test_gradual_policy_transition(self):
    policy_mgr = AdaptivePolicyManager()
    old_policy = RiskPolicy(max_position=1000)
    new_policy = RiskPolicy(max_position=2000)
    
    transition = policy_mgr.create_transition(old_policy, new_policy, duration=60)
    # Verify intermediate states during transition
    intermediate = transition.get_state_at(30)  # 30 seconds in
    assert 1000 < intermediate.max_position < 2000
```

---

**Prompt 10: Add Retry with Exponential Backoff for API Calls**

**Role & Context:**
- You are a **Senior API Integration Engineer**. The `data_fetcher.py` lacks proper handling for API rate limits and temporary failures.

**Task / Objective:**
- Implement retry mechanism with exponential backoff, jitter, and circuit breaking for all external API calls.
- **Update tests/ files related to the patch**: Add retry behavior tests, rate limit handling tests, and circuit breaker integration tests in `tests/integration/test_api_retry_mechanisms.py`.

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
# Test case to add:
def test_exponential_backoff_retry(self):
    fetcher = DataFetcher(retry_strategy=ExponentialBackoff(max_retries=3))
    mock_session = Mock()
    mock_session.get.side_effect = [RateLimitError, RateLimitError, "success"]
    
    result = await fetcher.fetch_data("test_endpoint")
    assert result == "success"
    assert mock_session.get.call_count == 3
```

---

Here are the updated prompts with the "update tests/ files related to the patch" detail added to each one:

---

**Prompt 11: Fix Unbounded Memory Growth in Historical Data Loading**

**Role & Context:**
- You are a **Senior Data Engineer**. The `historical_loader.py` suffers from unbounded memory growth when processing large datasets, risking system crashes.

**Task / Objective:**
- Implement streaming data processing with chunked loading and memory-efficient data structures for historical data operations.
- **Update tests/ files related to the patch**: Add memory usage monitoring tests, chunked processing validation, and large dataset handling tests in `tests/data/test_memory_efficient_loading.py`.

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
# Test case to add:
def test_chunked_loading_memory_efficiency(self):
    loader = HistoricalDataLoader(max_memory_mb=100)
    large_dataset = generate_large_dataset(size_gb=5)
    # Should process without exceeding memory limits
    chunks = list(loader.load_chunked(large_dataset))
    assert len(chunks) > 0
    assert get_memory_usage() < 100 * 1024 * 1024  # Under 100MB
```

---

**Prompt 12: Add Version Locking for Dataset Updates**

**Role & Context:**
- You are a **Senior Database Architect**. The `dataset_versioning.py` has race conditions during concurrent version updates that could corrupt datasets.

**Task / Objective:**
- Implement distributed locking mechanism for dataset version operations to prevent concurrent modification conflicts.
- **Update tests/ files related to the patch**: Add concurrent update tests, lock timeout tests, and deadlock detection tests in `tests/data/test_dataset_versioning_locks.py`.

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
# Test case to add:
def test_concurrent_version_updates_prevented(self):
    version_mgr = DatasetVersionManager()
    dataset_id = "test_dataset"
    
    def update_version():
        version_mgr.update_version(dataset_id, "v2")
    
    # Attempt concurrent updates
    threads = [threading.Thread(target=update_version) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    # Only one update should succeed
    assert version_mgr.get_version(dataset_id) in ["v1", "v2"]
```

---

**Prompt 13: Add Real-Time Model Performance Tracking**

**Role & Context:**
- You are a **Senior MLOps Engineer**. The `model_monitor.py` has delays in model drift detection, risking trading with stale models.

**Task / Objective:**
- Implement real-time performance tracking with streaming metrics and immediate drift alerts for ML models.
- **Update tests/ files related to the patch**: Add real-time drift detection tests, streaming metrics accuracy tests, and alert timing tests in `tests/ml/test_realtime_model_monitoring.py`.

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
# Test case to add:
def test_realtime_drift_detection_latency(self):
    monitor = RealTimeModelMonitor()
    start_time = time.time()
    # Simulate rapid model degradation
    for i in range(100):
        monitor.record_prediction(features=[i], prediction=0.5, actual=0.1)
    detection_time = monitor.last_drift_alert_time
    assert detection_time - start_time < 5.0  # Should detect within 5 seconds
```

---

**Prompt 14: Ensure Deterministic Optimization Results**

**Role & Context:**
- You are a **Senior Research Engineer**. The `genetic_optimizer.py` produces non-deterministic results due to random seed issues, making backtests irreproducible.

**Task / Objective:**
- Implement strict random seed control and deterministic execution for all optimization algorithms.
- **Update tests/ files related to the patch**: Add reproducibility tests, seed isolation tests, and deterministic benchmark tests in `tests/optimization/test_deterministic_optimization.py`.

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
# Test case to add:
def test_optimization_reproducibility(self):
    optimizer = GeneticOptimizer()
    # Run with same seed should produce identical results
    result1 = optimizer.optimize(seed=42)
    result2 = optimizer.optimize(seed=42)
    assert result1.parameters == result2.parameters
    assert result1.fitness == result2.fitness
```

---

**Prompt 15: Vectorize Feature Calculations**

**Role & Context:**
- You are a **Senior Performance Optimization Engineer**. The `features.py` uses inefficient loop-based calculations that bottleneck feature engineering.

**Task / Objective:**
- Vectorize all feature calculations using NumPy/Pandas operations and eliminate Python loops for performance gains.
- **Update tests/ files related to the patch**: Add performance benchmark tests, numerical accuracy tests, and edge case handling tests in `tests/features/test_vectorized_operations.py`.

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
# Test case to add:
def test_vectorized_performance_improvement(self):
    feature_calc = FeatureCalculator()
    large_data = generate_large_price_series(100000)
    
    # Time vectorized vs loop-based
    start = time.time()
    vectorized_result = feature_calc.calculate_rsi_vectorized(large_data)
    vectorized_time = time.time() - start
    
    start = time.time()
    loop_result = feature_calc.calculate_rsi_loop(large_data)
    loop_time = time.time() - start
    
    assert vectorized_time < loop_time / 10  # 10x speedup
    assert np.allclose(vectorized_result, loop_result)  # Same results
```

---

**Prompt 16: Implement Comprehensive Order Validation Pipeline**

**Role & Context:**
- You are a **Senior Trading Systems Architect**. The framework lacks comprehensive order validation, risking invalid orders reaching exchanges.

**Task / Objective:**
- Implement multi-stage order validation pipeline with pre-trade checks, risk validation, and exchange compatibility verification.
- **Update tests/ files related to the patch**: Add validation stage tests, exchange-specific rule tests, and latency measurement tests in `tests/orders/test_comprehensive_validation.py`.

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
# Test case to add:
def test_validation_pipeline_stages(self):
    validator = OrderValidationPipeline()
    invalid_order = Order(symbol="BTC/USDT", quantity=0)  # Invalid quantity
    
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(invalid_order)
    
    assert "quantity" in str(exc_info.value).lower()
    assert validator.get_failed_stages() == ["basic_syntax"]
```

---

**Prompt 17: Add Distributed Circuit Breaker Coordination**

**Role & Context:**
- You are a **Senior Distributed Systems Architect**. Circuit breakers operate in isolation, risking inconsistent states across distributed components.

**Task / Objective:**
- Implement distributed circuit breaker coordination with consensus and state synchronization across all framework instances.
- **Update tests/ files related to the patch**: Add distributed state sync tests, network partition tests, and consensus algorithm tests in `tests/distributed/test_circuit_breaker_coordination.py`.

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
# Test case to add:
def test_distributed_circuit_breaker_consensus(self):
    breakers = [DistributedCircuitBreaker(node_id=i) for i in range(3)]
    # One breaker trips - others should sync
    breakers[0].trip()
    
    # Wait for state propagation
    time.sleep(0.1)
    assert all(b.state == "open" for b in breakers)
```

---

**Prompt 18: Enforce Strict Memory Limits**

**Role & Context:**
- You are a **Senior Systems Reliability Engineer**. The framework lacks strict memory limits, risking out-of-memory crashes during high load.

**Task / Objective:**
- Implement hard memory limits with proactive monitoring and graceful degradation when limits are approached.
- **Update tests/ files related to the patch**: Add memory limit enforcement tests, graceful degradation tests, and emergency cleanup tests in `tests/system/test_memory_limits.py`.

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
# Test case to add:
def test_memory_limit_enforcement(self):
    memory_guard = MemoryGuard(limit_mb=50)
    large_data = "x" * (40 * 1024 * 1024)  # 40MB
    
    with memory_guard:
        # This should work
        processed = process_data(large_data)
    
    # Attempt to exceed limit
    with pytest.raises(MemoryLimitExceeded):
        with memory_guard:
            huge_data = "x" * (60 * 1024 * 1024)  # 60MB
            process_data(huge_data)
```

---

**Prompt 19: Implement Comprehensive Error Recovery**

**Role & Context:**
- You are a **Senior Resilience Engineer**. The framework lacks comprehensive error recovery procedures, risking extended downtime after failures.

**Task / Objective:**
- Implement systematic error recovery with state restoration, cleanup, and resumption capabilities for all major components.
- **Update tests/ files related to the patch**: Add recovery procedure tests, state restoration tests, and failure scenario tests in `tests/system/test_error_recovery.py`.

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
# Test case to add:
def test_automated_recovery_after_crash(self):
    trading_engine = TradingEngine()
    # Simulate crash during trade execution
    with patch.object(trading_engine, '_execute', side_effect=RuntimeError("Crash")):
        with pytest.raises(RuntimeError):
            await trading_engine.execute_trade(test_signal)
    
    # System should recover automatically
    recovery_success = await trading_engine.recovery_manager.recover()
    assert recovery_success
    assert trading_engine.state == "ready"
```

---

**Prompt 20: Add Idempotent Message Processing**

**Role & Context:**
- You are a **Senior Message Systems Engineer**. The framework lacks idempotent message processing, risking duplicate processing of signals and orders.

**Task / Objective:**
- Implement idempotent message processing with deduplication, exactly-once semantics, and idempotency keys for all critical messages.
- **Update tests/ files related to the patch**: Add duplicate detection tests, idempotency guarantee tests, and message replay tests in `tests/messaging/test_idempotent_processing.py`.

**Constraints & Rules:**
- Must maintain message processing performance
- Cannot lose legitimate messages
- Support configurable deduplication windows
- Provide duplicate detection metrics

**Input & Expected Output Format:**
- **Input:** Current non-idempotent message processing
- **Output:** Idempotent message processing system + updated test files

**Edge Cases & Additional Conditions:**
- Handle message replay attacks
- Support idempotency key generation and validation
- Provide message processing idempotency guarantees
- Integrate with existing message queues

**Examples / Snippets:**
```python
# Test case to add:
def test_exactly_once_processing_guarantee(self):
    processor = IdempotentMessageProcessor()
    message = TradingSignal(id="test_123", symbol="BTC/USDT")
    
    # Process same message multiple times
    result1 = await processor.process(message)
    result2 = await processor.process(message)
    result3 = await processor.process(message)
    
    # Should only execute once
    assert result1.executed
    assert not result2.executed  # Duplicate detected
    assert not result3.executed  # Duplicate detected
    assert processor.get_duplicate_count() == 2
```

---

All prompts now include specific test file update requirements, ensuring comprehensive test coverage for each critical patch while maintaining the framework's reliability and performance standards.