=============================================== short test summary info ===============================================
FAILED tests/api/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_latency_measurement - assert 5324.3 < 5000
FAILED tests/data/test_data_fixes.py::TestRefactoredFunctions::test_convert_to_dataframe - AttributeError: 'DataFetcher' object has no attribute '_convert_to_dataframe'
FAILED tests/data/test_memory_efficient_loading.py::TestMemoryEfficientLoading::test_large_dataset_loading_with_timeout - Failed: Large dataset loading timed out
FAILED tests/features/test_vectorized_operations.py::TestVectorizedOperations::test_vectorized_large_dataset_performance - assert 136.71458530426025 < 10.0
FAILED tests/features/test_vectorized_operations.py::TestVectorizedOperations::test_obv_edge_cases - assert nan == 100
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_retry_timeout_prevention - Failed: DID NOT RAISE (<class 'asyncio.exceptions.TimeoutError'>, <class 'Exception'>)
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_successful_retry_after_failures - assert not True
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_circuit_breaker_integration - AssertionError: assert False
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_endpoint_specific_retry_config - assert not True
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_graceful_degradation_on_api_unavailable - AssertionError: assert [] == ['open', 'hig...se', 'volume']
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_request_deduplication_for_idempotent_operations - assert not True
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_permanent_vs_temporary_failure_handling - Failed: DID NOT RAISE <class 'ccxt.base.errors.BadRequest'>
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_circuit_breaker_recovery - AssertionError: assert 'half-open' == 'closed'
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_retry_metrics_and_logging - assert not True
FAILED tests/integration/test_api_retry_mechanisms.py::TestAPIRetryMechanisms::test_configurable_retry_strategies - assert not True
FAILED tests/integration/test_cross_feature_integration.py::TestCircuitBreakerMonitoringIntegration::test_monitoring_performance_during_circuit_breaker - AssertionError: Performance degraded by 97.1% during circuit breaker (threshold: 50%)
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_python_random - assert [0.2186379748...06032167, ...] == [0.6394267984...74229113, ...]
FAILED tests/optimization/test_deterministic_optimization.py::TestDeterministicOptimization::test_optimization_timeout_safety - assert 79.4978494644165 < 30
18 failed, 3368 passed, 100 skipped, 1194 warnings in 1587.57s (0:26:27)