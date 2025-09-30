=============================================== short test summary info ===============================================
FAILED tests/test_integration_benchmark.py::TestBenchmarkExceptionHandling::test_benchmark_operation_with_exception_relaxed_mode - RuntimeError: Error on call 1
FAILED tests/test_integration_benchmark.py::TestBenchmarkExceptionHandling::test_benchmark_mixed_success_failure - ConnectionError: Network timeout
FAILED tests/test_integration_benchmark.py::TestBenchmarkExceptionHandling::test_benchmark_all_failures_relaxed_mode - Exception: Always fails
FAILED tests/test_integration_benchmark.py::TestBenchmarkExceptionHandling::test_benchmark_different_exception_types - ValueError: Value error
FAILED tests/test_integration_benchmark.py::TestBenchmarkIntegration::test_run_performance_benchmark_relaxed_mode - ValueError: Occasional failure
FAILED tests/test_order_idempotency.py::TestOrderIdempotency::test_order_with_empty_idempotency_key_raises_error - Failed: DID NOT RAISE <class 'core.exceptions.MissingIdempotencyError'>
FAILED tests/test_order_idempotency.py::TestOrderIdempotency::test_different_idempotency_keys_allow_multiple_executions - AssertionError: assert {'id': 'order...us': 'filled'} == {'id': 'order...us': 'filled'}
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_initialization - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_policy_selection_small_order - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_policy_selection_large_order - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_policy_selection_high_spread - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_policy_selection_stable_liquidity - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_simple_order_execution - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_validation_failure - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_retry_success - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_fallback - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_adaptive_pricing_applied - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_result_structure - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_get_execution_status - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_get_active_executions - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_cancel_execution - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_default_config - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_small_limit_order_flow - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_large_order_twap_flow - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
ERROR tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_high_spread_dca_flow - AttributeError: 'ConfigManager' object has no attribute 'get_reliability_config'
=================== 70 failed, 3023 passed, 20 skipped, 2 warnings, 18 errors in 553.69s (0:09:13) ====================