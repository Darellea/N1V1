=============================================== short test summary info ===============================================
FAILED tests/core/test_logging_and_resources.py::TestStructuredLogging::test_structured_message_formatting - assert 'component=test' in '{"message": "Test message", "component": "test", "operation": "test_op", "user_id": "12...
FAILED tests/core/test_monitoring_observability.py::TestMetricsEndpoint::test_metrics_endpoint_response - AssertionError: assert 401 == 200
FAILED tests/core/test_monitoring_observability.py::TestMetricsEndpoint::test_health_endpoint - AssertionError: assert 401 == 200
FAILED tests/core/test_monitoring_observability.py::TestMetricsEndpoint::test_invalid_endpoint - AssertionError: assert 401 == 404
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_with_validation_failure - assert None is not None
FAILED tests/integration/test_cross_feature_integration.py::TestCircuitBreakerMonitoringIntegration::test_monitoring_performance_during_circuit_breaker - AssertionError: Performance degraded by 264.6% during circuit breaker (threshold: 50%)
FAILED tests/integration/test_cross_feature_integration.py::TestPerformanceCircuitBreakerIntegration::test_performance_metrics_circuit_breaker_impact - AssertionError: Circuit breaker caused 52.0% performance impact
FAILED tests/security/test_secure_logging.py::TestStructuredLogger::test_structured_warning_logging - assert 'WARNING' in '{"message": "Alert triggered", "alert_type": "high_cpu", "threshold": "***AMOUNT_MASKED***"}\n'
FAILED tests/security/test_secure_logging.py::TestStructuredLogger::test_structured_error_logging - assert 'ERROR' in '{"message": "Database connection failed", "db_host": "localhost", "error_code": "***AMOUNT_MASKE...
========================= 9 failed, 2915 passed, 20 skipped, 2 warnings in 739.50s (0:12:19) ==========================