=============================================== short test summary info ===============================================
FAILED tests/api/test_api_app.py::TestCustomExceptionMiddleware::test_normal_request_passthrough - AssertionError: Expected 'mock' to have been called once. Called 0 times.
FAILED tests/core/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_baseline_coefficient_variation_edge_cases - assert False
FAILED tests/core/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_anomaly_detection_zero_std - TypeError: object of type 'coroutine' has no len()
FAILED tests/core/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_percentile_anomaly_score_calculation - TypeError: object of type 'coroutine' has no len()
FAILED tests/core/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_decimal_initial_balance_conversion - AssertionError: assert False
FAILED tests/core/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_invalid_initial_balance_fallback - AssertionError: assert False
FAILED tests/core/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_none_initial_balance_fallback - AssertionError: assert False
FAILED tests/core/test_algorithmic_correctness.py::TestStatisticalEdgeCases::test_percentile_calculation_robustness - assert 4.8 >= 4.96
FAILED tests/core/test_binary_integration.py::TestBinaryModelIntegrationEdgeCases::test_binary_model_prediction_with_exact_threshold - assert False == True
FAILED tests/core/test_binary_integration.py::TestBinaryModelIntegrationEdgeCases::test_binary_model_prediction_with_low_confidence - assert -1.0 == 0.55
FAILED tests/core/test_binary_integration.py::TestGlobalIntegrationFunctions::test_integrate_binary_model_convenience_function - NameError: name 'integrate_binary_model' is not defined
FAILED tests/core/test_binary_integration.py::TestErrorRecovery::test_binary_model_prediction_error_recovery - assert -1.0 == 0.0
FAILED tests/core/test_binary_integration.py::TestErrorRecovery::test_strategy_selection_error_recovery - AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == 'UNKNOWN'
FAILED tests/core/test_binary_integration.py::TestErrorRecovery::test_complete_pipeline_error_recovery - AssertionError: assert 'Integration error' in 'Binary integration disabled'
FAILED tests/core/test_binary_integration.py::TestPerformanceAndScalability::test_concurrent_market_data_processing - AssertionError: assert False == True
FAILED tests/core/test_binary_integration.py::TestLoggingAndMonitoring::test_binary_prediction_logging - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/core/test_binary_integration_enhanced.py::TestBinaryModelIntegrationFeatureExtraction::test_calculate_macd_edge_cases - assert 0.02243589743588359 == 0.0
FAILED tests/core/test_binary_integration_enhanced.py::TestBinaryModelIntegrationGlobalFunctions::test_get_binary_integration_with_config - assert False == True
FAILED tests/core/test_binary_integration_enhanced.py::TestBinaryModelIntegrationMetricsIntegration::test_metrics_recording_in_prediction - AssertionError: Expected 'record_prediction' to have been called once. Called 0 times.
FAILED tests/core/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_predict_binary_model_with_model_exception - assert -1.0 == 0.0
FAILED tests/core/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_select_strategy_with_regime_detector_failure - AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == <MarketRegime.UNKNOWN: 'unknown'>
FAILED tests/core/test_binary_integration_enhanced.py::TestBinaryModelIntegrationLogging::test_binary_prediction_logging_integration - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/core/test_binary_model_metrics.py::TestMetricsCollection::test_collect_binary_model_metrics_with_exception - AssertionError: assert False
FAILED tests/core/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_trade_frequency_drift - AssertionError: assert 'trade frequency changed' in 'Trade frequency changed by 100.0%'
FAILED tests/core/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_accuracy_drop - assert 0 == 1
FAILED tests/core/test_binary_model_metrics.py::TestPerformanceReporting::test_get_performance_report - KeyError: 'was_correct'
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_collect_metrics_with_binary_integration_import_error - AttributeError: <module 'core.binary_model_metrics' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\binary_m...
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_record_prediction_with_invalid_data - TypeError: '>=' not supported between instances of 'str' and 'float'
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_record_decision_outcome_with_invalid_data - TypeError: unsupported operand type(s) for +=: 'float' and 'str'
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_calculate_metrics_with_corrupted_data - TypeError: '>' not supported between instances of 'str' and 'datetime.datetime'
FAILED tests/core/test_binary_model_metrics.py::TestDataIntegrity::test_prediction_history_data_integrity - TypeError: BinaryModelMetricsCollector.record_prediction() got an unexpected keyword argument 'decision'
FAILED tests/core/test_bot_engine.py::TestBotEngine::test_trading_cycle - AssertionError: Expected 'get_historical_data' to have been called once. Called 0 times.
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_minimal_config - AssertionError: assert [] == ['BTC/USDT']
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_portfolio_mode - AssertionError: assert set() == {'BTC/USDT', 'ETH/USDT'}
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_invalid_balance - assert 10000.0 == 1000.0
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_empty_markets - AssertionError: assert [] == ['BTC/USDT']
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_calculate_max_drawdown - assert 0.061224489795918366 == 0.08
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_success - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_missing_equity - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineStateManagement::test_run_main_loop_normal_operation - AssertionError: Expected '_update_display' to have been called once. Called 0 times.
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_process_binary_integration_enabled - assert 0 == 1
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_execute_integrated_decisions - AssertionError: Expected 'execute_order' to have been called once. Called 0 times.
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_trading_cycle_with_data_fetch_error - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_signal_generation_with_strategy_error - Exception: Strategy failed
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_risk_evaluation_with_component_error - AttributeError: 'NoneType' object has no attribute 'evaluate_signal'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_order_execution_with_component_error - AttributeError: 'NoneType' object has no attribute 'execute_order'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_state_update_with_component_errors - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_empty_market_data - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_none_market_data - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_concurrent_trading_cycles - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_cache_comprehensive.py::TestCacheBatchOperations::test_set_multiple_ohlcv_success - assert False == True
FAILED tests/core/test_cache_comprehensive.py::TestCacheInvalidation::test_invalidate_symbol_data_specific_types - assert 3 == 1
FAILED tests/core/test_cache_comprehensive.py::TestCacheContextManager::test_context_manager_success - RuntimeError: Cache not initialized and no config provided
FAILED tests/core/test_cache_comprehensive.py::TestCacheErrorHandling::test_batch_operations_with_partial_failures - AssertionError: assert 'ETH/USDT' in {'ADA/USDT': None, 'BTC/USDT': {'price': 50000}}
FAILED tests/core/test_cache_eviction.py::TestCacheEviction::test_evict_expired_entries - AssertionError: Expected 'delete' to have been called.
FAILED tests/core/test_cache_eviction.py::TestCacheIntegration::test_memory_manager_integration - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/core/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_valid_config - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_missing_sections - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_component_factory_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_cache_component_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_memory_manager_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_component_caching - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_configuration_override - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_async_component_operations - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_factory_global_instance - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/core/test_dependency_injection.py::TestConfigurationIntegration::test_performance_tracker_config_integration - assert 1000.0 == 2000.0
FAILED tests/core/test_dependency_injection.py::TestConfigurationPersistence::test_config_save_load - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/core/test_logging_and_resources.py::TestStructuredLogging::test_logger_initialization - AssertionError: assert <LogSensitivity.SECURE: 'secure'> == <LogSensitivity.INFO: 'info'>
FAILED tests/core/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_context_manager - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_initialization_failure - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_close_error_handling - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_context_manager_with_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_global_instance_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not hav...
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_endpoint_global_instance_cleanup - TypeError: object NoneType can't be used in 'await' expression
FAILED tests/core/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cleanup_on_startup_failure - assert <Application 0x219f8033d90> is None
FAILED tests/core/test_monitoring_observability.py::TestTradingMetricsCollection::test_trading_metrics_collection - TypeError: object numpy.float64 can't be used in 'await' expression
FAILED tests/core/test_monitoring_observability.py::TestAlertingSystem::test_alert_deduplication - assert not True
FAILED tests/core/test_monitoring_observability.py::TestAlertingSystem::test_notification_delivery - AssertionError: Expected '_send_discord_notification' to have been called once. Called 0 times.
FAILED tests/core/test_order_manager.py::TestOrderManager::test_init_paper_mode - assert <core.order_manager.MockLiveExecutor object at 0x00000219FC0A6140> is None
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
FAILED tests/security/test_core_security.py::TestDataProcessorSecurity::test_calculate_rsi_batch_data_validation - AssertionError: assert 'TEST' not in {'TEST':    open\n0   100\n1   101\n2   102}
FAILED tests/security/test_core_security.py::TestMetricsEndpointSecurity::test_secure_defaults - assert False == True
FAILED tests/security/test_secure_logging.py::TestLogSanitizer::test_api_key_sanitization - assert '***API_KEY_MASKED***' in 'Using API key: "sk-1234567890abcdef" for authentication'
FAILED tests/security/test_secure_logging.py::TestLogSanitizer::test_financial_amount_sanitization - AssertionError: assert '***BALANCE_MASKED***' in '***12345.67_MASKED***, ***-987.65_MASKED***, ***11234.56_MASKED***'
FAILED tests/security/test_secure_logging.py::TestLogSanitizer::test_audit_level_minimal_output - AssertionError: assert 'Security eve...y sk-123 used' == '[AUDIT] Secu... event logged'
FAILED tests/security/test_secure_logging.py::TestLogSanitizer::test_complex_message_sanitization - assert '***API_KEY_MASKED***' in '\n        Processing trade for ***EMAIL_MASKED*** with API key "sk-1234567890abcd...
FAILED tests/security/test_secure_logging.py::TestStructuredLogger::test_structured_info_logging - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/security/test_secure_logging.py::TestStructuredLogger::test_structured_warning_logging - AssertionError: assert 'WARNING' in 'Alert triggered | alert_type=high_cpu | threshold=90.5\n'
FAILED tests/security/test_secure_logging.py::TestStructuredLogger::test_structured_error_logging - AssertionError: assert 'ERROR' in 'Database connection failed | db_host=localhost | error_code=500\n'
FAILED tests/security/test_secure_logging.py::TestStructuredLogger::test_debug_level_preserves_data - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/security/test_secure_logging.py::TestStructuredLogger::test_sensitivity_change - AssertionError: assert '***API_KEY_MASKED***' in 'Test message | api_key=sk-test123\n'
FAILED tests/security/test_secure_logging.py::TestCoreLoggerIntegration::test_core_logger_sanitization - AssertionError: Expected 'info' to have been called once. Called 0 times.
FAILED tests/security/test_secure_logging.py::TestLogSecurityVerification::test_no_api_keys_in_logs - AssertionError: assert 'sk-1234567890abcdef' not in 'Using API k...567890abcdef'
FAILED tests/security/test_secure_logging.py::TestLogSecurityVerification::test_no_financial_amounts_in_logs - AssertionError: assert '12345.67' not in '***12345.67_MASKED***'
FAILED tests/security/test_secure_logging.py::TestLogSecurityVerification::test_log_structure_preservation - AssertionError: assert '***AMOUNT_MASKED***' in 'Processing trade for ***EMAIL_MASKED*** with amount 12345.67 and A...
ERROR tests/api/test_api_app.py::TestRateLimitJSONMiddleware::test_rate_limit_response_conversion - ImportError: cannot import name 'RateLimitJSONMiddleware' from 'api.app' (C:\Users\TU\Desktop\new project\N1V1\api\...
ERROR tests/api/test_api_app.py::TestRateLimitJSONMiddleware::test_normal_response_passthrough - ImportError: cannot import name 'RateLimitJSONMiddleware' from 'api.app' (C:\Users\TU\Desktop\new project\N1V1\api\...
ERROR tests/core/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_constant_returns - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/core/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_single_return - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/core/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_profit_factor_edge_cases - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/core/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_total_return_percentage_safe_division - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/core/test_cache_eviction.py::TestCacheIntegration::test_end_to_end_eviction_workflow
================== 125 failed, 2863 passed, 20 skipped, 107 warnings, 7 errors in 806.27s (0:13:26) ===================