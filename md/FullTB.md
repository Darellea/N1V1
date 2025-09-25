=============================================== short test summary info ===============================================
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_missing_equity - assert 10000.0 == 10050.0
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_execute_integrated_decisions - AssertionError: Expected 'execute_order' to have been called once. Called 0 times.
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_state_update_with_component_errors - assert 0.0 == 10000.0
FAILED tests/core/test_cache_comprehensive.py::TestCacheBatchOperations::test_set_multiple_ohlcv_success - assert False == True
FAILED tests/core/test_cache_comprehensive.py::TestCacheInvalidation::test_invalidate_symbol_data_specific_types - assert 3 == 1
FAILED tests/core/test_cache_comprehensive.py::TestCacheContextManager::test_context_manager_success - RuntimeError: Cache not initialized and no config provided
FAILED tests/core/test_cache_comprehensive.py::TestCacheErrorHandling::test_batch_operations_with_partial_failures - AssertionError: assert 'ETH/USDT' in {'ADA/USDT': None, 'BTC/USDT': {'price': 50000}}
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_memory_thresholds_check - AttributeError: 'dict' object has no attribute 'max_memory_mb'
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_evict_expired_entries - AssertionError: Expected 'delete' to have been called.
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_enforce_cache_limits_ttl_policy - AssertionError: Expected '_evict_oldest' to be called once. Called 0 times.
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_enforce_cache_limits_lru_policy - AssertionError: Expected '_evict_lru' to be called once. Called 0 times.
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_enforce_cache_limits_lfu_policy - AssertionError: Expected '_evict_lfu' to be called once. Called 0 times.
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_perform_maintenance_normal_conditions - KeyError: 'maintenance_performed'
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_perform_maintenance_high_memory - KeyError: 'maintenance_performed'
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_get_cache_stats - KeyError: 'cache_size'
FAILED tests/core/test_cache_comprehensive.py::TestCacheEviction::test_eviction_under_load - KeyError: 'maintenance_performed'
FAILED tests/core/test_cache_comprehensive.py::TestCacheIntegration::test_memory_manager_integration - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_config_manager.py::TestConfigManagerValidation::test_validate_main_config_unknown_field - assert 0 > 0
FAILED tests/core/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_valid_config - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_missing_sections - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_unknown_fields - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_config_manager.py::TestPydanticModels::test_main_config_unknown_field - Failed: DID NOT RAISE <class 'pydantic_core._pydantic_core.ValidationError'>
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_component_factory_creation - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_cache_component_creation - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_memory_manager_creation - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_component_caching - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_configuration_override - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/core/test_dependency_injection.py::TestConfigurationIntegration::test_performance_tracker_config_integration - assert 1000.0 == 2000.0
FAILED tests/core/test_dependency_injection.py::TestConfigurationPersistence::test_config_save_load - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/core/test_logging_and_resources.py::TestStructuredLogging::test_logger_initialization - AssertionError: assert <LogSensitivity.SECURE: 'secure'> == <LogSensitivity.INFO: 'info'>
FAILED tests/core/test_logging_and_resources.py::TestStructuredLogging::test_structured_message_formatting - assert 'component=test' in '{"level": "INFO", "message": "Test message", "component": "test", "operation": "test_op...
FAILED tests/core/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_context_manager - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_initialization_failure - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_close_error_handling - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_context_manager_with_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_global_instance_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_endpoint_global_instance_cleanup - TypeError: object NoneType can't be used in 'await' expression
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cleanup_on_startup_failure - assert <Application 0x2ac6a059c90> is None
FAILED tests/core/test_monitoring_observability.py::TestTradingMetricsCollection::test_trading_metrics_collection - TypeError: object numpy.float64 can't be used in 'await' expression
FAILED tests/core/test_monitoring_observability.py::TestMetricsEndpoint::test_metrics_endpoint_response - AssertionError: assert 401 == 200
FAILED tests/core/test_monitoring_observability.py::TestMetricsEndpoint::test_health_endpoint - AssertionError: assert 401 == 200
FAILED tests/core/test_monitoring_observability.py::TestMetricsEndpoint::test_invalid_endpoint - AssertionError: assert 401 == 404
FAILED tests/core/test_monitoring_observability.py::TestAlertingSystem::test_alert_deduplication - assert not True
FAILED tests/core/test_monitoring_observability.py::TestAlertingSystem::test_notification_delivery - AssertionError: Expected '_send_discord_notification' to have been called once. Called 0 times.
FAILED tests/core/test_order_manager.py::TestOrderManager::test_init_paper_mode - assert <core.order_manager.MockLiveExecutor object at 0x000002AC689BC670> is None
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_paper_mode - assert None is not None
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_backtest_mode - assert None is not None
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_active - assert None is not None
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_live_mode_with_retry - assert None is not None
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_trigger_counter - AssertionError: assert False
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_unknown_mode - AssertionError: Expected 'execute_paper_order' to be called once. Called 0 times.
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_valid - ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_invalid_symbol_format - AssertionError: Regex pattern did not match.
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_negative_amount - AssertionError: Regex pattern did not match.
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_stop_without_loss - AssertionError: Regex pattern did not match.
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_invalid_signal_order_combo - AssertionError: Regex pattern did not match.
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_with_valid_payload - assert None is not None
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_buy_signal - AssertionError: assert Decimal('50050.0000') == Decimal('50025')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_sell_signal - AssertionError: assert Decimal('5994.0000') == Decimal('5997')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_portfolio_mode_buy - AssertionError: assert Decimal('1.200050') < Decimal('0.1')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_buy_order - AssertionError: assert Decimal('50050.000') == Decimal('50025')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_sell_order - AssertionError: assert Decimal('49950.000') == Decimal('49975')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_with_dict_signal - AssertionError: assert Decimal('30030.000') == Decimal('30015')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_different_order_types - AssertionError: assert <OrderType.MARKET: 'market'> == <OrderType.LIMIT: 'limit'>
FAILED tests/core/test_performance_optimization.py::TestPerformanceMonitoringIntegration::test_monitoring_during_optimization - TypeError: object float can't be used in 'await' expression
FAILED tests/core/test_performance_optimization.py::TestPerformanceMonitoringIntegration::test_anomaly_detection_with_optimizations - TypeError: object float can't be used in 'await' expression
FAILED tests/core/test_regression.py::TestRegression::test_network_error_retry_mechanism - AssertionError: Expected 'retry_async' to have been called once. Called 0 times.
FAILED tests/core/test_regression.py::TestRegression::test_exchange_error_handling - AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
FAILED tests/core/test_regression.py::TestRegression::test_timeout_error_recovery - AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
FAILED tests/core/test_regression.py::TestRegression::test_memory_leak_prevention_in_long_running_sessions - assert False
FAILED tests/core/test_regression.py::TestRegression::test_invalid_trading_mode_fallback - assert None is not None
FAILED tests/core/test_signal_router.py::TestJournalWriter::test_append_with_event_loop - AssertionError: Expected 'create_task' to have been called once. Called 0 times.
FAILED tests/core/test_signal_router.py::TestJournalWriter::test_stop_method - AssertionError: Expected mock to have been awaited once. Awaited 0 times.
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_fallback - AssertionError: assert False
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_adaptive_pricing_applied - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_small_limit_order_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_large_order_twap_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_high_spread_dca_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/integration/test_cross_feature_integration.py::TestCircuitBreakerMonitoringIntegration::test_monitoring_alerts_during_circuit_breaker - TypeError: object float can't be used in 'await' expression
FAILED tests/integration/test_cross_feature_integration.py::TestMonitoringPerformanceIntegration::test_performance_metrics_monitoring_system - TypeError: object float can't be used in 'await' expression
FAILED tests/integration/test_cross_feature_integration.py::TestMonitoringPerformanceIntegration::test_monitoring_performance_profiling_operations - TypeError: object float can't be used in 'await' expression
FAILED tests/integration/test_cross_feature_integration.py::TestMonitoringPerformanceIntegration::test_resource_usage_monitoring_profiling - TypeError: object float can't be used in 'await' expression
FAILED tests/integration/test_cross_feature_integration.py::TestFullSystemIntegration::test_end_to_end_trading_scenario - TypeError: object float can't be used in 'await' expression
FAILED tests/integration/test_cross_feature_integration.py::TestFullSystemIntegration::test_stress_test_all_features - TypeError: object float can't be used in 'await' expression
FAILED tests/integration/test_cross_feature_integration.py::TestFullSystemIntegration::test_failure_recovery_integration - TypeError: object float can't be used in 'await' expression
FAILED tests/integration/test_cross_feature_integration.py::TestFullSystemIntegration::test_performance_regression_detection - TypeError: object float can't be used in 'await' expression
ERROR tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_initialize_performance_stats
ERROR tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_update_pnl
ERROR tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_update_win_loss_counts
ERROR tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_log_status
ERROR tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_print_status_table_exception
ERROR tests/core/test_cache_comprehensive.py::TestCacheIntegration::test_end_to_end_eviction_workflow
==================== 86 failed, 2832 passed, 20 skipped, 2 warnings, 6 errors in 771.15s (0:12:51) ====================