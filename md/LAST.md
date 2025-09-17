============================================== short test summary info ==============================================
FAILED demo/test_anomaly_integration.py::test_basic_anomaly_detection - RuntimeError: no running event loop
FAILED demo/test_anomaly_integration.py::test_risk_manager_integration - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/api/test_endpoints.py::TestDashboardEndpoint::test_dashboard_endpoint_returns_200 - assert 500 == 200    
FAILED tests/api/test_endpoints.py::TestDashboardEndpoint::test_dashboard_endpoint_returns_html - AssertionError: assert 'text/html' in 'application/json'
FAILED tests/api/test_endpoints.py::TestRateLimiting::test_dashboard_endpoint_not_rate_limited - assert 500 == 200    
FAILED tests/api/test_endpoints.py::TestCORSSecurity::test_cors_allows_configured_origins - AssertionError: assert 'access-control-allow-origin' in Headers({'content-length': '48', 'content-type': 'applica...
FAILED tests/api/test_endpoints.py::TestCustomExceptionMiddleware::test_custom_exception_middleware_handles_exceptions - assert False
FAILED tests/api/test_endpoints.py::TestTemplateRendering::test_dashboard_template_not_found - assert 500 == 200      
FAILED tests/api/test_endpoints.py::TestPrometheusMetrics::test_api_requests_counter_incremented - assert 429 == 200  
FAILED tests/api/test_endpoints.py::TestRateLimitingEdgeCases::test_get_remote_address_exempt_function - AttributeError: 'MockRequest' object has no attribute 'client'
FAILED tests/api/test_endpoints.py::TestMiddlewareOrder::test_cors_middleware_configured - assert False
FAILED tests/api/test_endpoints.py::TestMiddlewareOrder::test_rate_limit_middleware_configured - assert False
FAILED tests/core/test_async_optimizer.py::TestThreadAndProcessOperations::test_run_in_process_success - AttributeError: Can't pickle local object 'TestThreadAndProcessOperations.test_run_in_process_success.<locals>.cp...
FAILED tests/core/test_async_optimizer.py::TestThreadAndProcessOperations::test_run_in_process_with_exception - AttributeError: Can't pickle local object 'TestThreadAndProcessOperations.test_run_in_process_with_exception.<loc...        
FAILED tests/core/test_async_optimizer.py::TestHealthChecks::test_health_check_success - AttributeError: 'ProcessPoolExecutor' object has no attribute '_shutdown'. Did you mean: 'shutdown'?
FAILED tests/core/test_async_optimizer.py::TestHealthChecks::test_health_check_thread_pool_failure - AttributeError: 'ProcessPoolExecutor' object has no attribute '_shutdown'. Did you mean: 'shutdown'?
FAILED tests/core/test_async_optimizer.py::TestShutdownAndCleanup::test_shutdown_success - AttributeError: 'ProcessPoolExecutor' object has no attribute '_shutdown'. Did you mean: 'shutdown'?
FAILED tests/core/test_async_optimizer.py::TestShutdownAndCleanup::test_shutdown_with_pending_tasks - AssertionError: 
assert (False or False)
FAILED tests/core/test_async_optimizer.py::TestErrorHandling::test_async_file_write_error_handling - FileNotFoundError: [WinError 3] The system cannot find the path specified: ''
FAILED tests/core/test_async_optimizer.py::TestEdgeCases::test_memory_stats_size_limit - KeyError: 'memory_usage'     
FAILED tests/core/test_binary_model_metrics.py::TestMetricsCollection::test_collect_binary_model_metrics_with_exception - AssertionError: assert False
FAILED tests/core/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_trade_frequency_drift - AssertionError: assert 'trade frequency changed' in 'Trade frequency changed by 100.0%'
FAILED tests/core/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_accuracy_drop - assert 0 == 1
FAILED tests/core/test_binary_model_metrics.py::TestPerformanceReporting::test_get_performance_report - KeyError: 'was_correct'
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_collect_metrics_with_binary_integration_import_error - AttributeError: <module 'core.binary_model_metrics' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\binary...
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_record_prediction_with_invalid_data - TypeError: '>=' not supported between instances of 'str' and 'float'
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_record_decision_outcome_with_invalid_data - TypeError: unsupported operand type(s) for +=: 'float' and 'str'
FAILED tests/core/test_binary_model_metrics.py::TestErrorHandling::test_calculate_metrics_with_corrupted_data - TypeError: '>' not supported between instances of 'str' and 'datetime.datetime'
FAILED tests/core/test_binary_model_metrics.py::TestDataIntegrity::test_prediction_history_data_integrity - TypeError: BinaryModelMetricsCollector.record_prediction() got an unexpected keyword argument 'decision'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_minimal_config - AssertionError: assert [] == ['BTC/USDT']
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_portfolio_mode - 
AssertionError: assert set() == {'BTC/USDT', 'ETH/USDT'}
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_empty_markets - AssertionError: assert [] == ['BTC/USDT']
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineTradingCycle::test_generate_signals_from_strategies - AttributeError: BUY
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineTradingCycle::test_evaluate_risk_approved - AttributeError: BUY
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineTradingCycle::test_evaluate_risk_rejected - AttributeError: BUY
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineTradingCycle::test_execute_orders_success - AttributeError: BUY
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_calculate_max_drawdown - assert 0.061224489795918366 == 0.08
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_calculate_sharpe_ratio - NameError: name 'np' is not defined
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_success - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_missing_equity - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineStateManagement::test_run_main_loop_normal_operation 
- AssertionError: Expected '_update_display' to have been called once. Called 0 times.
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_process_binary_integration_enabled 
- assert 0 == 1
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_execute_integrated_decisions - AssertionError: Expected 'execute_order' to have been called once. Called 0 times.
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_trading_cycle_with_data_fetch_error - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_signal_generation_with_strategy_error - Exception: Strategy failed
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_risk_evaluation_with_component_error - AttributeError: 'NoneType' object has no attribute 'evaluate_signal'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_order_execution_with_component_error - AttributeError: 'NoneType' object has no attribute 'execute_order'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_state_update_with_component_errors - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_empty_market_data 
- AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_none_market_data - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_concurrent_trading_cycles - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_extract_multi_tf_data_with_valid_data - KeyError: 'environment'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_extract_multi_tf_data_with_missing_data - KeyError: 'environment'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_extract_multi_tf_data_with_invalid_symbol - KeyError: 'environment'
FAILED tests/core/test_bot_engine_comprehensive.py::TestBotEngineUtilityFunctions::test_combine_market_data_function - KeyError: 'environment'
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_fallback - AssertionError: assert False
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_adaptive_pricing_applied - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_small_limit_order_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_large_order_twap_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_high_spread_dca_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_normal - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required ...
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_high_volatility - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required ... 
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_empty_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_none_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_market_monitor_error_handling - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required ...
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_position_sizing_with_adaptive_policy - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_policy_called_with_correct_data - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_kill_switch_blocks_trading - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_defensive_mode_reduces_position_size - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_atr_position_sizing_with_multiplier - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_error_handling_in_adaptive_integration - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_different_position_sizing_methods_with_adaptive - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'      
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_multiplier_application - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerPerformanceIntegration::test_trade_outcome_updates_adaptive_policy - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerPerformanceIntegration::test_consecutive_losses_affect_position_sizing - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_calculation - assert FalseFAILED tests/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_fallback - AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x00000245365...
FAILED tests/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_position_sizing_method_selection - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_error_handling_in_calculations - TypeError: '>=' 
not supported between instances of 'coroutine' and 'int'
FAILED tests/test_adaptive_risk.py::TestAdaptiveRiskIntegration::test_complete_trade_workflow - assert False is True  
FAILED tests/test_alert_rules_manager.py::TestAlertRuleEdgeCases::test_rule_evaluation_with_non_numeric_metric - TypeError: '>' not supported between instances of 'str' and 'float'
FAILED tests/test_alert_rules_manager.py::TestAlertRulesManagerEdgeCases::test_get_alert_history_with_limit_zero - AssertionError: assert [{'rule_name'...stamp': 1000}] == []
FAILED tests/test_alert_rules_manager.py::TestConfigurationValidation::test_config_with_invalid_deduplication_window - AssertionError: assert 'invalid' == 300
FAILED tests/test_alert_rules_manager.py::TestPerformanceScenarios::test_manager_with_large_history - AssertionError: 
assert 200 == 10000
FAILED tests/test_alert_rules_manager.py::TestConcurrency::test_concurrent_rule_evaluation - AssertionError: assert 9 
== 10
FAILED tests/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_constant_returns - 
assert 3222.525096876663 == 0.0
FAILED tests/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_profit_factor_edge_cases - KeyError: 'profit_factor'
FAILED tests/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_baseline_coefficient_variation_edge_cases - assert False
FAILED tests/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_anomaly_detection_zero_std - TypeError: object of type 'coroutine' has no len()
FAILED tests/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_percentile_anomaly_score_calculation - TypeError: object of type 'coroutine' has no len()
FAILED tests/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_decimal_initial_balance_conversion - AssertionError: assert False
FAILED tests/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_invalid_initial_balance_fallback - AssertionError: assert False
FAILED tests/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_none_initial_balance_fallback - AssertionError: assert False
FAILED tests/test_algorithmic_correctness.py::TestStatisticalEdgeCases::test_percentile_calculation_robustness - assert 4.8 >= 4.96
FAILED tests/test_anomaly_detector.py::TestPriceGapDetector::test_detect_normal_price_change - AssertionError: assert 
False is False
FAILED tests/test_anomaly_detector.py::TestPriceGapDetector::test_detect_price_gap - AssertionError: assert True is True
FAILED tests/test_anomaly_detector.py::TestPriceGapDetector::test_detect_small_gap - AssertionError: assert False is False
FAILED tests/test_anomaly_detector.py::TestAnomalyDetector::test_initialization_with_config - KeyError: 'scale_down_threshold'
FAILED tests/test_anomaly_detector.py::TestAnomalyDetector::test_check_signal_anomaly_skip_trade - RuntimeError: no running event loop
FAILED tests/test_anomaly_detector.py::TestAnomalyDetector::test_check_signal_anomaly_scale_down - RuntimeError: no running event loop
FAILED tests/test_anomaly_detector.py::TestAnomalyDetector::test_get_anomaly_statistics - AttributeError: 'AnomalyDetector' object has no attribute 'get_anomaly_statistics'
FAILED tests/test_anomaly_detector.py::TestAnomalyDetector::test_empty_statistics - AttributeError: 'AnomalyDetector' 
object has no attribute 'get_anomaly_statistics'
FAILED tests/test_anomaly_detector.py::TestAnomalyLogging::test_log_to_file - KeyError: 'enabled'
FAILED tests/test_anomaly_detector.py::TestAnomalyLogging::test_log_to_json - KeyError: 'enabled'
FAILED tests/test_anomaly_detector.py::TestAnomalyLogging::test_trade_logger_integration - AttributeError: 'AnomalyDetector' object has no attribute '_log_anomaly'. Did you mean: 'log_anomalies'?
FAILED tests/test_anomaly_detector.py::TestEdgeCases::test_extreme_values - AssertionError: assert True is True       
FAILED tests/test_anomaly_detector.py::TestConfiguration::test_custom_configuration - KeyError: 'skip_trade_threshold'FAILED tests/test_anomaly_detector.py::TestConfiguration::test_severity_threshold_conversion - KeyError: 'scale_down_factor'
FAILED tests/test_asset_selector.py::TestAssetSelector::test_fetch_from_coingecko_async_success - AssertionError: assert None == {'ETH/USDT': 50000000000}
FAILED tests/test_backtester.py::TestRegimeAwareMetrics::test_regime_aware_with_mismatched_lengths - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 2 recor...
FAILED tests/test_backtester.py::TestRegimeAwareMetrics::test_regime_aware_with_insufficient_data - AssertionError: assert 'per_regime_metrics' in {'overall': {'equity_curve': [100.0], 'losses': 0, 'max_drawdown':...
FAILED tests/test_backtester.py::TestPerformanceOptimizations::test_vectorized_returns_calculation - NameError: name 'isfinite' is not defined
FAILED tests/test_backtester.py::TestPerformanceOptimizations::test_pandas_regime_grouping_performance - AssertionError: assert 'per_regime_metrics' in {'overall': {'equity_curve': [1000.0, 1001.0, 1002.0, 1003.0, 1004...
FAILED tests/test_backtester.py::TestValidationFunctions::test_validate_equity_progression_with_invalid_types - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestValidationError'>
FAILED tests/test_backtester.py::TestRegimeAwareFunctions::test_compute_regime_aware_metrics_with_no_regime_data - KeyError: 'total_regimes'
FAILED tests/test_backtester.py::TestRegimeAwareFunctions::test_align_regime_data_lengths_with_mismatch - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 2 recor...
FAILED tests/test_backtester.py::TestHelperFunctions::test_export_regime_csv_summary - AssertionError: assert 'bull' in 'Regime,Total Return,Sharpe Ratio,Win Rate,Max Drawdown,Total Trades,Avg Confide...
FAILED tests/test_backtester.py::TestErrorHandling::test_export_equity_progression_with_io_error - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestSecurityError'>
FAILED tests/test_backtester.py::TestErrorHandling::test_compute_backtest_metrics_with_invalid_data - numpy.core._exceptions._UFuncNoLoopError: ufunc 'maximum' did not contain a loop with signature matching types (d...
FAILED tests/test_backtester.py::TestFileOperations::test_export_with_permission_denied - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestSecurityError'>
FAILED tests/test_binary_integration.py::TestBinaryModelIntegrationEdgeCases::test_binary_model_prediction_with_exact_threshold - assert False == True
FAILED tests/test_binary_integration.py::TestBinaryModelIntegrationEdgeCases::test_binary_model_prediction_with_low_confidence - assert -1.0 == 0.55
FAILED tests/test_binary_integration.py::TestGlobalIntegrationFunctions::test_integrate_binary_model_convenience_function - NameError: name 'integrate_binary_model' is not defined
FAILED tests/test_binary_integration.py::TestErrorRecovery::test_binary_model_prediction_error_recovery - assert -1.0 
== 0.0
FAILED tests/test_binary_integration.py::TestErrorRecovery::test_strategy_selection_error_recovery - AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == 'UNKNOWN'
FAILED tests/test_binary_integration.py::TestErrorRecovery::test_complete_pipeline_error_recovery - AssertionError: assert 'Integration error' in 'Binary integration disabled'
FAILED tests/test_binary_integration.py::TestPerformanceAndScalability::test_concurrent_market_data_processing - AssertionError: assert False == True
FAILED tests/test_binary_integration.py::TestLoggingAndMonitoring::test_binary_prediction_logging - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationFeatureExtraction::test_calculate_macd_edge_cases - assert 0.02243589743588359 == 0.0
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationGlobalFunctions::test_get_binary_integration_with_config - assert False == True
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationMetricsIntegration::test_metrics_recording_in_prediction - AssertionError: Expected 'record_prediction' to have been called once. Called 0 times.
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationMetricsIntegration::test_metrics_recording_in_decision - AttributeError: BULLISH
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_predict_binary_model_with_model_exception - assert -1.0 == 0.0
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_validate_risk_with_component_failure - AttributeError: BULLISH
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationLogging::test_binary_prediction_logging_integration - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/test_binary_integration_enhanced.py::TestBinaryModelIntegrationLogging::test_decision_outcome_logging_integration - AttributeError: BULLISH
FAILED tests/test_cache_comprehensive.py::TestCacheSpecializedMethods::test_set_market_ticker - KeyError: 'default'   
FAILED tests/test_cache_comprehensive.py::TestCacheSpecializedMethods::test_set_ohlcv - KeyError: 'default'
FAILED tests/test_cache_comprehensive.py::TestCacheSpecializedMethods::test_set_account_balance - KeyError: 'default' 
FAILED tests/test_cache_comprehensive.py::TestCacheBatchOperations::test_set_multiple_ohlcv_success - KeyError: 'default'
FAILED tests/test_cache_comprehensive.py::TestCacheBatchOperations::test_set_multiple_ohlcv_failure - KeyError: 'default'
FAILED tests/test_cache_comprehensive.py::TestCacheInvalidation::test_invalidate_symbol_data_all_types - AssertionError: expected call not found.
FAILED tests/test_cache_comprehensive.py::TestCacheInvalidation::test_invalidate_symbol_data_specific_types - assert 2 == 1
FAILED tests/test_cache_comprehensive.py::TestCacheContextManager::test_context_manager_success - RuntimeError: Cache 
not initialized and no config provided
FAILED tests/test_cache_comprehensive.py::TestCacheErrorHandling::test_batch_operations_with_partial_failures - KeyError: 'ETH/USDT'
FAILED tests/test_cache_comprehensive.py::TestCacheMemoryMonitoring::test_memory_thresholds - AttributeError: 'CacheConfig' object has no attribute 'memory_config'
FAILED tests/test_cache_comprehensive.py::TestCacheMemoryMonitoring::test_get_cache_stats - KeyError: 'cache_size'    
FAILED tests/test_cache_eviction.py::TestCacheEviction::test_cache_config_defaults - AttributeError: 'CacheConfig' object has no attribute 'memory_config'
FAILED tests/test_core_security.py::TestDataProcessorSecurity::test_calculate_rsi_batch_data_validation - AssertionError: assert 'TEST' not in {'TEST':    open\n0   100\n1   101\n2   102}
FAILED tests/test_core_security.py::TestMetricsEndpointSecurity::test_secure_defaults - assert False == True
FAILED tests/test_cross_asset_validation.py::TestAssetSelector::test_apply_market_cap_weighting - assert 0.5 > 0.5    
FAILED tests/test_data.py::test_cache_operations - AssertionError: assert False
FAILED tests/test_data_fetcher.py::TestDataFetcherInitialization::test_init_with_config - AssertionError: assert 'C:\\Users\\T...\\.test_cache' == '.test_cache'
FAILED tests/test_data_fetcher.py::TestDataFetcherInitialization::test_init_cache_directory_creation - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
FAILED tests/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_with_caching - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
FAILED tests/test_data_fetcher.py::TestDataFetcherErrorScenarios::test_cache_operations_error_handling - data.data_fetcher.PathTraversalError: Invalid cache directory path: /invalid/path
FAILED tests/test_data_fixes.py::TestDataFetcherTimestampHandling::test_critical_cache_raises_exception_on_failure - AssertionError: CacheLoadError not raised
FAILED tests/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_all_strategies_fail - AssertionError: <coroutine object DataFetcher._load_from_cache at 0x0000024536ACF760> is not None
FAILED tests/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_1_integer_ms - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_2_datetime_column - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_3_format_parsing - 
AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timezone_normalization - TypeError: Object of 
type Timestamp is not JSON serializable
FAILED tests/test_data_fixes.py::TestDatasetVersioningHashFix::test_deterministic_sampling_large_dataset - NameError: 
name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestDatasetVersioningHashFix::test_deterministic_sampling_small_dataset - NameError: 
name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestDatasetVersioningHashFix::test_full_hash_option - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestDatasetVersioningHashFix::test_hash_changes_with_data_changes - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestGapHandlingStrategies::test_forward_fill_strategy_with_gaps - NameError: name 'np' is not defined
FAILED tests/test_data_fixes.py::TestGapHandlingStrategies::test_interpolation_strategy - NameError: name 'np' is not 
defined
FAILED tests/test_data_fixes.py::TestGapHandlingStrategies::test_reject_strategy_with_gaps - NameError: name 'np' is not defined
FAILED tests/test_data_fixes.py::TestGapHandlingStrategies::test_unknown_gap_strategy_defaults_to_forward_fill - NameError: name 'np' is not defined
FAILED tests/test_data_fixes.py::TestDataOptimizations::test_cache_save_optimization - NameError: name 'np' is not defined
FAILED tests/test_data_fixes.py::TestDataOptimizations::test_concatenation_generator_optimization - NameError: name 'np' is not defined
FAILED tests/test_data_fixes.py::TestRefactoredFunctions::test_convert_to_dataframe - AssertionError: Lists differ: ['open', 'high', 'low', 'close', 'volume'] != ['timestamp', 'open', 'high', 'low', ...
FAILED tests/test_data_fixes.py::TestRefactoredFunctions::test_exchange_wrapper_properties - AttributeError: 'DataFetcher' object has no attribute '_ExchangeWrapper'
FAILED tests/test_data_fixes.py::TestStructuredLogging::test_data_fetcher_structured_logging - AssertionError: Expected 'info' to have been called.
FAILED tests/test_data_fixes.py::TestStructuredLogging::test_historical_loader_structured_logging - AssertionError: Expected 'info' to have been called.
FAILED tests/test_data_fixes.py::TestMetadataErrorHandling::test_attempt_backup_recovery_no_backup_file - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestMetadataErrorHandling::test_corrupted_metadata_json_error - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestMetadataErrorHandling::test_corrupted_metadata_with_backup_recovery - NameError: 
name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestMetadataErrorHandling::test_metadata_backup_recovery_failure - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestMetadataErrorHandling::test_metadata_file_not_found - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestMetadataErrorHandling::test_successful_metadata_load - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestDatasetVersioningStructuredLogging::test_create_version_structured_logging - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_fixes.py::TestDatasetVersioningStructuredLogging::test_migrate_legacy_dataset_structured_logging - NameError: name 'DatasetVersionManager' is not defined
FAILED tests/test_data_module_refactoring.py::TestPagination::test_execute_pagination_loop_with_data - assert 2 == 1  
FAILED tests/test_data_module_refactoring.py::TestPagination::test_infinite_loop_detection - assert True is False     
FAILED tests/test_data_module_refactoring.py::TestPathTraversal::test_sanitize_cache_path_valid - KeyError: 'name'    
FAILED tests/test_data_module_refactoring.py::TestPathTraversal::test_sanitize_cache_path_traversal - KeyError: 'name'FAILED tests/test_data_module_refactoring.py::TestBackwardCompatibility::test_all_public_methods_still_exist - NameError: name 'time' is not defined
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_valid_relative - KeyError: 'name'
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_traversal_dots - KeyError: 'name'
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_absolute_path - KeyError: 'name'
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_backslash_traversal - KeyError: 'name'
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_complex_traversal - KeyError: 'name'
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_valid_nested - KeyError: 'name'
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_empty_string - KeyError: 'name'
FAILED tests/test_data_security.py::TestPathTraversalPrevention::test_cache_disabled_no_validation - KeyError: 'name' 
FAILED tests/test_data_security.py::TestDatasetVersionManagerSecurity::test_create_version_with_validation - NameError: name 'time' is not defined
FAILED tests/test_data_security.py::TestDatasetVersionManagerSecurity::test_create_version_with_invalid_dataframe - NameError: name 'time' is not defined
FAILED tests/test_data_security.py::TestDatasetVersionManagerSecurity::test_migrate_legacy_dataset_validation - AttributeError: <module 'data.dataset_versioning' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\data\\dataset_...        
FAILED tests/test_data_security.py::TestDatasetVersionManagerSecurity::test_migrate_legacy_dataset_invalid_input - NameError: name 'time' is not defined
FAILED tests/test_data_security.py::TestVersionNameSanitization::test_sanitize_version_name_absolute_paths - AssertionError: Regex pattern did not match.
FAILED tests/test_data_security.py::TestVersionNameSanitization::test_sanitize_version_name_invalid_characters - AssertionError: Regex pattern did not match.
FAILED tests/test_data_security.py::TestVersionNameSanitization::test_create_version_with_malicious_name - NameError: 
name 'time' is not defined
FAILED tests/test_data_security.py::TestVersionNameSanitization::test_create_version_with_valid_name - NameError: name 'time' is not defined
FAILED tests/test_data_security.py::TestConfigurationValidation::test_validate_data_directory_absolute_paths - AssertionError: Regex pattern did not match.
FAILED tests/test_data_security.py::TestConfigurationValidation::test_validate_data_directory_invalid_characters - AssertionError: Regex pattern did not match.
FAILED tests/test_data_security.py::TestConfigurationValidation::test_setup_data_directory_with_valid_config - AssertionError: assert False
FAILED tests/test_data_security.py::TestIntegrationSecurity::test_full_data_pipeline_security - NameError: name 'time' is not defined
FAILED tests/test_data_security.py::TestIntegrationSecurity::test_error_handling_and_logging - KeyError: 'name'       
FAILED tests/test_dependency_injection.py::TestDependencyInjection::test_component_factory_creation - assert False    
FAILED tests/test_dependency_injection.py::TestDependencyInjection::test_cache_component_creation - assert False      
FAILED tests/test_dependency_injection.py::TestDependencyInjection::test_memory_manager_creation - assert False       
FAILED tests/test_dependency_injection.py::TestDependencyInjection::test_configuration_override - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/test_dependency_injection.py::TestConfigurationIntegration::test_performance_tracker_config_integration - assert 1000.0 == 2000.0
FAILED tests/test_dependency_injection.py::TestConfigurationPersistence::test_config_save_load - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/test_discord_notifier.py::TestDiscordNotifier::test_send_signal_alert - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_discord_notifier.py::TestDiscordNotifier::test_send_signal_alert_disabled_alerts - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_extract_confidence_default_from_strength - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_majority_vote_no_consensus - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_weighted_vote_high_weight_dominates - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_confidence_average_high_confidence_buy - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_confidence_average_low_confidence_no_trade - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_majority_vote_tie_handling - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_weighted_vote_with_zero_weights - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_confidence_average_empty_confidences - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_extract_confidence_invalid_values - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleManager::test_strategy_signal_dataclass - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_ensemble_manager.py::TestEnsembleIntegration::test_ensemble_with_multiple_strategies - assert None is not None
FAILED tests/test_features.py::TestFeatureExtractor::test_extract_features_insufficient_data - AttributeError: <module 'ml.features' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\features.py'> does not...
FAILED tests/test_features.py::TestFeatureExtractor::test_extract_features_success - AttributeError: <module 'ml.features' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\features.py'> does not...
FAILED tests/test_features.py::TestFeatureExtractor::test_extract_features_invalid_data - AttributeError: <module 'ml.features' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\features.py'> does not...
FAILED tests/test_historical_loader.py::TestHistoricalDataLoaderInitialization::test_init_with_config - AssertionError: assert 'C:\\Users\\T...storical_data' == 'test_historical_data'
FAILED tests/test_historical_loader.py::TestHistoricalDataLoaderInitialization::test_init_default_values - AssertionError: assert 'C:\\Users\\T...storical_data' == 'historical_data'
FAILED tests/test_historical_loader.py::TestHistoricalDataLoaderInitialization::test_setup_data_directory - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Tem...
FAILED tests/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_historical_data_success - NameError: name 'time' is not defined
FAILED tests/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_historical_data_partial_failure - NameError: name 'time' is not defined
FAILED tests/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_symbol_data_from_cache - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Tem...       
FAILED tests/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_symbol_data_force_refresh - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Tem...    
FAILED tests/test_integration.py::TestOptimizationIntegration::test_output_validation_backtest_metrics - AssertionError: assert False
FAILED tests/test_integration.py::TestOptimizationIntegration::test_error_scenario_optimization_with_invalid_parameters - AttributeError: 'GeneticOptimizer' object has no attribute 'ParameterBounds'. Did you mean: 'parameter_bounds'?   
FAILED tests/test_knowledge_base.py::TestStorageBackends::test_json_storage - PermissionError: Path traversal attempt 
detected: C:\Users\TU\AppData\Local\Temp\tmp8k175f1j\test_knowledge.json
FAILED tests/test_knowledge_base.py::TestStorageBackends::test_csv_storage - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpjbo20rde\test_knowledge.csv
FAILED tests/test_knowledge_base.py::TestStorageBackends::test_sqlite_storage - AssertionError: assert None
FAILED tests/test_knowledge_base.py::TestAdaptiveWeighting::test_market_similarity_calculation - AttributeError: 'AdaptiveWeightingEngine' object has no attribute '_calculate_market_similarity'
FAILED tests/test_knowledge_base.py::TestAdaptiveWeighting::test_performance_score_calculation - AttributeError: 'AdaptiveWeightingEngine' object has no attribute '_calculate_performance_score'
FAILED tests/test_knowledge_base.py::TestAdaptiveWeighting::test_update_knowledge_from_trade - assert None
FAILED tests/test_knowledge_base.py::TestKnowledgeManager::test_knowledge_manager_creation - AttributeError: 'KnowledgeManager' object has no attribute 'storage'
FAILED tests/test_knowledge_base.py::TestKnowledgeManager::test_store_trade_knowledge - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpe6us5liy\knowledge.json
FAILED tests/test_knowledge_base.py::TestKnowledgeManager::test_get_adaptive_weights - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmppkqb8e98\knowledge.json
FAILED tests/test_knowledge_base.py::TestKnowledgeManager::test_knowledge_statistics - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmp_mvab3wc\knowledge.json
FAILED tests/test_knowledge_base.py::TestKnowledgeManager::test_disabled_knowledge_manager - AttributeError: 'KnowledgeManager' object has no attribute 'storage'
FAILED tests/test_knowledge_base.py::TestWeightingCalculator::test_calculate_strategy_weight_with_knowledge - TypeError: unhashable type: 'MarketCondition'
FAILED tests/test_knowledge_base.py::TestCacheManager::test_cache_manager_initialization - assert <knowledge_base.adaptive.LRUCache object at 0x0000024537790040> == {}
FAILED tests/test_knowledge_base.py::TestKnowledgeValidator::test_validate_query_parameters_valid - TypeError: KnowledgeQuery.__init__() got an unexpected keyword argument 'max_confidence'
FAILED tests/test_knowledge_base.py::TestKnowledgeValidator::test_validate_query_parameters_invalid - TypeError: KnowledgeQuery.__init__() got an unexpected keyword argument 'max_confidence'
FAILED tests/test_knowledge_base.py::TestDataStoreInterface::test_save_entry - AssertionError: assert <MagicMock name='save_entry()' id='2496300661440'> is True
FAILED tests/test_knowledge_base.py::TestDataStoreInterface::test_get_entry - AssertionError: Expected 'get_entry' to 
be called once. Called 0 times.
FAILED tests/test_knowledge_base.py::TestDataStoreInterface::test_query_entries - AssertionError: Expected 'query_entries' to be called once. Called 0 times.
FAILED tests/test_knowledge_base.py::TestDataStoreInterface::test_list_entries - AssertionError: Expected 'list_entries' to be called once. Called 0 times.
FAILED tests/test_knowledge_base.py::TestDataStoreInterface::test_get_stats - AssertionError: Expected 'get_stats' to 
have been called once. Called 0 times.
FAILED tests/test_knowledge_base.py::TestEdgeCases::test_corrupted_storage_file - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpp2nm0q7f\corrupted.json
FAILED tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_limit_order - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_sell_signal - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_with_params - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_context_manager_with_cleanup - 
AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not h...  
FAILED tests/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_global_instance_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not h...       
FAILED tests/test_logging_and_resources.py::TestResourceCleanupIntegration::test_endpoint_global_instance_cleanup - TypeError: object NoneType can't be used in 'await' expression
FAILED tests/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cleanup_on_startup_failure - assert <Application 0x2453b4a6320> is None
FAILED tests/test_ml.py::TestTrainFunctions::test_load_historical_data_csv - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/test_ml.py::TestTrainFunctions::test_prepare_training_data - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', '...
FAILED tests/test_ml.py::TestTrainFunctions::test_prepare_training_data_insufficient_samples - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', '...
FAILED tests/test_ml.py::TestTrainerFunctions::test_validate_inputs - TypeError: validate_inputs() missing 1 required 
positional argument: 'feature_columns'
FAILED tests/test_ml.py::TestTrainerFunctions::test_validate_inputs_missing_label - TypeError: validate_inputs() missing 1 required positional argument: 'feature_columns'
FAILED tests/test_ml.py::TestModelLoaderFunctions::test_predict - TypeError: 'Mock' object is not iterable
FAILED tests/test_ml.py::TestConfiguration::test_calculate_all_indicators - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/test_ml.py::TestConfiguration::test_calculate_bollinger_bands - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/test_ml.py::TestConfiguration::test_calculate_ema - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/test_ml.py::TestConfiguration::test_calculate_macd - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/test_ml.py::TestConfiguration::test_calculate_rsi - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/test_ml.py::TestConfiguration::test_insufficient_data_rsi - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/test_ml.py::TestConfiguration::test_validate_ohlcv_data - AttributeError: 'TestConfiguration' object has 
no attribute 'test_data'
FAILED tests/test_ml.py::TestIntegration::test_full_pipeline - ValueError: Training data cannot be empty
FAILED tests/test_ml_artifact_model_card.py::test_train_creates_model_card - NameError: name 'ProcessPoolExecutor' is 
not defined
FAILED tests/test_ml_filter.py::TestMLFilter::test_save_load_model - assert False
FAILED tests/test_ml_filter.py::TestFactoryFunctions::test_load_ml_filter - assert False
FAILED tests/test_monitoring_observability.py::TestMetricsCollection::test_metric_recording - AssertionError: assert 0 == 1
FAILED tests/test_monitoring_observability.py::TestMetricsCollection::test_histogram_observation - AssertionError: assert 0 == 3
FAILED tests/test_monitoring_observability.py::TestDataAccuracy::test_timestamp_accuracy - IndexError: list index out 
of range
FAILED tests/test_monitoring_observability.py::TestDataAccuracy::test_concurrent_metric_recording - assert 0 == 500   
FAILED tests/test_monitoring_observability.py::TestMetricsEndpoint::test_metrics_endpoint_response - AssertionError: assert 401 == 200
FAILED tests/test_monitoring_observability.py::TestMetricsEndpoint::test_health_endpoint - AssertionError: assert 401 
== 200
FAILED tests/test_monitoring_observability.py::TestMetricsEndpoint::test_invalid_endpoint - AssertionError: assert 401 == 404
FAILED tests/test_monitoring_observability.py::TestAlertingSystem::test_alert_deduplication - assert not True
FAILED tests/test_monitoring_observability.py::TestAlertingSystem::test_notification_delivery - AssertionError: Expected '_send_discord_notification' to have been called once. Called 0 times.
FAILED tests/test_optimization.py::TestWalkForwardOptimizer::test_optimization_with_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/test_optimization.py::TestGeneticOptimizer::test_population_initialization - AssertionError: assert 20 == 10
FAILED tests/test_optimization.py::TestGeneticOptimizer::test_population_initialization_no_bounds - assert 20 == 5    
FAILED tests/test_optimization.py::TestCoreOptimizationAlgorithms::test_crossover_with_different_gene_counts - KeyError: 'param2'
FAILED tests/test_optimization.py::TestCoreOptimizationAlgorithms::test_parameter_validation_edge_cases - AssertionError: assert not True
FAILED tests/test_optimization.py::TestRLOptimizer::test_policy_save_load - assert 0 > 0
FAILED tests/test_optimization.py::TestOptimizerFactory::test_create_optimizer - assert 20 == 10
FAILED tests/test_optimization.py::TestCrossPairValidation::test_cross_pair_validation_insufficient_data - NameError: 
name 'start_time' is not defined
FAILED tests/test_order_manager.py::TestOrderManager::test_execute_order_paper_mode - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_order_manager.py::TestOrderManager::test_execute_order_backtest_mode - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_active - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_order_manager.py::TestOrderManager::test_execute_order_live_mode_with_retry - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_trigger_counter - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_order_manager.py::TestOrderManager::test_execute_order_unknown_mode - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_no_regime_data - KeyError: 'total_return'
FAILED tests/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_with_regime_data - KeyError: 'total_trades'
FAILED tests/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_missing_regime_data - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 3 recor...
FAILED tests/test_regime_aware_backtester.py::TestRegimeAwareExport::test_export_regime_aware_report - KeyError: 'report_type'
FAILED tests/test_regime_aware_backtester.py::TestEdgeCases::test_regime_data_longer_than_equity - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 1 recor...
FAILED tests/test_regime_aware_backtester.py::TestEdgeCases::test_regime_data_with_errors - KeyError: 'trade_count'   
FAILED tests/test_regime_aware_backtester.py::TestEdgeCases::test_insufficient_data_regime - KeyError: 'avg_confidence'
FAILED tests/test_regression.py::TestRegression::test_network_error_retry_mechanism - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_exchange_error_handling - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_timeout_error_recovery - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_corrupted_signal_data_handling - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_discord_notification_failure_does_not_break_flow - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_discord_rate_limit_handling - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_memory_leak_prevention_in_long_running_sessions - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_invalid_trading_mode_fallback - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_regression.py::TestRegression::test_extreme_market_conditions_simulation - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_risk.py::TestAnomalyDetection::test_price_zscore_detector_anomalous_data - assert False
FAILED tests/test_risk.py::TestAnomalyDetection::test_volume_zscore_detector_anomalous_volume - assert 0 > 0
FAILED tests/test_risk.py::TestIntegrationTests::test_full_risk_assessment_workflow - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_risk.py::TestIntegrationTests::test_anomaly_detection_integration - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_risk.py::TestExtremeConditions::test_flash_crash_simulation - AssertionError: assert <AnomalyType.PRICE_ZSCORE: 'price_zscore'> in {<AnomalyType.PRICE_GAP: 'price_gap'>}
FAILED tests/test_risk.py::TestExtremeConditions::test_extreme_volume_spike - assert 0 > 0
FAILED tests/test_risk.py::TestRiskManagerEdgeCases::test_risk_manager_zero_balance - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_risk.py::TestRiskManagerEdgeCases::test_risk_manager_extreme_price_values - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_risk.py::TestRiskManagerEdgeCases::test_invalid_signal_handling - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_secure_logging.py::TestLogSanitizer::test_api_key_sanitization - assert '***API_KEY_MASKED***' in 'Using API key: "sk-1234567890abcdef" for authentication'
FAILED tests/test_secure_logging.py::TestLogSanitizer::test_secret_token_sanitization - AssertionError: assert '***TOKEN_MASKED***' in 'Token: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
FAILED tests/test_secure_logging.py::TestLogSanitizer::test_audit_level_minimal_output - AssertionError: assert 'Security eve...y sk-123 used' == '[AUDIT] Secu... event logged'
FAILED tests/test_secure_logging.py::TestLogSanitizer::test_complex_message_sanitization - assert '***EMAIL_MASKED***' in '\n        Processing trade for user@example.com with API key "sk-1234567890abcdef...
FAILED tests/test_secure_logging.py::TestStructuredLogger::test_structured_info_logging - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/test_secure_logging.py::TestStructuredLogger::test_structured_warning_logging - AssertionError: assert 'WARNING' in 'Alert triggered | alert_type=high_cpu | threshold=90.5\n'
FAILED tests/test_secure_logging.py::TestStructuredLogger::test_structured_error_logging - AssertionError: assert 'ERROR' in 'Database connection failed | db_host=localhost | error_code=500\n'
FAILED tests/test_secure_logging.py::TestStructuredLogger::test_debug_level_preserves_data - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/test_secure_logging.py::TestStructuredLogger::test_sensitivity_change - AssertionError: assert '***API_KEY_MASKED***' in 'Test message | api_key=sk-test123\n'
FAILED tests/test_secure_logging.py::TestCoreLoggerIntegration::test_core_logger_sanitization - AssertionError: Expected 'info' to have been called once. Called 0 times.
FAILED tests/test_secure_logging.py::TestGlobalSensitivity::test_global_sensitivity_change - AssertionError: assert <LogSensitivity.INFO: 'info'> == <LogSensitivity.DEBUG: 'debug'>
FAILED tests/test_secure_logging.py::TestLogSecurityVerification::test_no_api_keys_in_logs - AssertionError: assert 'sk-1234567890abcdef' not in 'Using API k...567890abcdef'
FAILED tests/test_secure_logging.py::TestLogSecurityVerification::test_no_financial_amounts_in_logs - AssertionError: 
assert '$999,999.99' not in 'Balance: $999,999.99'
FAILED tests/test_secure_logging.py::TestLogSecurityVerification::test_no_personal_info_in_logs - AssertionError: assert 'user@example.com' not in 'User data: ...@example.com'
FAILED tests/test_secure_logging.py::TestLogSecurityVerification::test_log_structure_preservation - AssertionError: assert '***EMAIL_MASKED***' in 'Processing trade for user@example.com with amount 12345.67 and AP...
FAILED tests/test_signal_router.py::test_process_signal_invalid_missing_symbol - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_invalid_zero_amount - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_validate_signal_invalid - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_signal_conflicts_opposite_entry - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_signal_conflicts_entry_vs_exit - TypeError: TradingSignal.__init__() missing 
1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_concurrent_signal_processing - TypeError: TradingSignal.__init__() missing 1 
required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_update_signal_status_nonexistent_signal - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_get_signal_history_limit - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_invalid_symbol_lock - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_resolve_conflicts_with_empty_list - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_resolve_conflicts_strength_based - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_resolve_conflicts_newer_first - assert None is not None
FAILED tests/test_signal_router.py::test_store_signal_with_journal_disabled - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_cancel_signal_with_journal_disabled - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_ml_extract_features_with_list_of_scalars - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_ml_extract_features_with_invalid_data - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_missing_candle_data - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_incomplete_ohlcv_sequence - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_empty_market_data - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_none_market_data - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_exchange_api_timeout - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_retry_on_api_timeout - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_persistent_api_failure - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_zero_liquidity - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_zero_volume - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_wide_spread - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_malformed_signal_data - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_missing_required_fields - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_extreme_values - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_unicode_in_fields - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_signal_router.py::test_process_signal_with_special_characters_in_metadata - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_strategies.py::TestBollingerReversionStrategy::test_generate_signals_oversold_reversion - assert 0 == 1
FAILED tests/test_strategies.py::TestBollingerReversionStrategy::test_generate_signals_overbought_reversion - assert 0 == 1
FAILED tests/test_strategies.py::TestMockingAndIntegration::test_signal_generation_deterministic_timestamps - assert 0 == 1
FAILED tests/test_strategy.py::test_strategy_lifecycle - assert 0 == 1
FAILED tests/test_strategy.py::test_signal_generation - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_strategy.py::test_strategy_run_method - assert 0 == 1
FAILED tests/test_strategy.py::test_strategy_performance_tracking - assert 0 == 3
FAILED tests/test_strategy_generator.py::TestStrategyGenerator::test_strategy_generation - assert None is not None    
FAILED tests/test_strategy_generator.py::TestPerformance::test_generation_performance - assert None is not None       
FAILED tests/test_strategy_integration.py::TestStrategyIntegrationWorkflow::test_strategy_error_handling_workflow - assert 0 >= 2
FAILED tests/test_strategy_integration.py::TestStrategyRiskManagerIntegration::test_risk_manager_position_sizing_integration - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/test_strategy_integration.py::TestStrategyDataFetcherIntegration::test_data_fetcher_error_recovery_integration - assert 1 == 2
FAILED tests/test_trading_signal_amount.py::test_fraction_to_notional - TypeError: TradingSignal.__init__() missing 1 
required positional argument: 'timestamp'
FAILED tests/test_trading_signal_amount.py::test_no_metadata_no_change - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_trading_signal_amount.py::test_missing_total_balance_raises - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_trading_signal_amount.py::test_clamp_fraction_above_one - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_trading_signal_amount.py::test_string_amount_conversion - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_trading_signal_amount.py::test_negative_fraction_clamped - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
FAILED tests/test_train.py::TestLoadHistoricalData::test_load_csv_data_success - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/test_train.py::TestLoadHistoricalData::test_load_data_with_symbol_filter - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/test_train.py::TestLoadHistoricalData::test_load_data_missing_required_columns - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/test_train.py::TestPrepareTrainingData::test_prepare_data_success - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', '...
FAILED tests/test_train.py::TestPrepareTrainingData::test_prepare_data_insufficient_samples - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', '...
FAILED tests/test_train.py::TestPrepareTrainingData::test_prepare_data_with_nan_values - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', '...
FAILED tests/test_train.py::TestPrepareTrainingData::test_prepare_data_with_zero_prices - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', '...
FAILED tests/test_train.py::TestPrepareTrainingData::test_prepare_data_outlier_removal - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', '...
FAILED tests/test_train.py::TestMainFunction::test_main_predictive_models_disabled - AssertionError: Expected 'exit' to not have been called. Called 1 times.
FAILED tests/test_trainer.py::TestDataProcessing::test_load_data_success - NameError: name 'logger' is not defined    
FAILED tests/test_trainer.py::TestDataProcessing::test_load_data_file_not_found - NameError: name 'logger' is not defined
FAILED tests/test_trainer.py::TestBinaryTraining::test_train_model_binary_basic - NameError: name 'ProcessPoolExecutor' is not defined
FAILED tests/test_trainer.py::TestBinaryTraining::test_train_model_binary_small_dataset - NameError: name 'ProcessPoolExecutor' is not defined
FAILED tests/test_trainer.py::TestBinaryTraining::test_train_model_binary_insufficient_data - NameError: name 'ProcessPoolExecutor' is not defined
FAILED tests/test_trainer.py::TestBinaryTraining::test_train_model_binary_with_tuning - NameError: name 'ProcessPoolExecutor' is not defined
FAILED tests/test_walk_forward.py::TestWalkForwardOptimizer::test_optimize_no_windows - NameError: name 'start_time' is not defined
FAILED tests/test_walk_forward.py::TestWalkForwardOptimizer::test_optimize_with_windows - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/test_walk_forward.py::TestIntegration::test_full_walk_forward_workflow - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/test_walk_forward.py::TestIntegration::test_error_handling_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/test_walk_forward.py::TestIntegration::test_error_handling_optimizer_failure - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
ERROR tests/core/test_async_optimizer.py::TestGlobalFunctions::test_async_read_file_convenience_function
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_memory_usage_monitoring - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_memory_thresholds_check - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_evict_expired_entries - TypeError: CacheConfig.__init__() 
got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_enforce_cache_limits_ttl_policy - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_enforce_cache_limits_lru_policy - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_enforce_cache_limits_lfu_policy - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_evict_lru - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_evict_lfu - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_evict_oldest - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_perform_maintenance_normal_conditions - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_perform_maintenance_high_memory - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_get_cache_stats - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_cache_not_connected - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_thread_safety - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_eviction_under_load - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheEviction::test_error_handling_in_eviction - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_cache_eviction.py::TestCacheIntegration::test_end_to_end_eviction_workflow
ERROR tests/test_data.py::test_historical_loader - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/test_data.py::test_historical_data_validation - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/test_data.py::test_data_resampling - data.historical_loader.ConfigurationError: data_dir contains invalid 
characters: .test_historical_data
ERROR tests/test_data.py::test_historical_data_pagination - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/test_data_fetcher.py::TestDataFetcherCaching::test_get_cache_key - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_empty - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_fetcher.py::TestDataFetcherCaching::test_save_and_load_from_cache - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_fetcher.py::TestDataFetcherCaching::test_save_to_cache_empty_dataframe - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_expired - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_fetcher.py::TestDataFetcherIntegration::test_full_workflow_with_caching - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_fetcher.py::TestDataFetcherIntegration::test_multiple_symbols_workflow - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_fetcher.py::TestDataFetcherIntegration::test_realtime_data_workflow - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Loc...
ERROR tests/test_data_module_refactoring.py::TestCaching::test_save_to_cache_success - KeyError: 'name'
ERROR tests/test_data_module_refactoring.py::TestCaching::test_load_from_cache_success - KeyError: 'name'
ERROR tests/test_data_module_refactoring.py::TestCaching::test_load_from_cache_nonexistent - KeyError: 'name'
ERROR tests/test_data_module_refactoring.py::TestCaching::test_load_from_cache_expired - KeyError: 'name'
ERROR tests/test_data_module_refactoring.py::TestCaching::test_cache_key_generation - KeyError: 'name'
ERROR tests/test_data_module_refactoring.py::TestVersioning::test_create_version_success - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/test_data_module_refactoring.py::TestVersioning::test_create_version_invalid_name - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/test_data_module_refactoring.py::TestVersioning::test_migrate_legacy_dataset - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_signal_alert - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
ERROR tests/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_full_data_loading_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Tem...
ERROR tests/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_data_validation_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Tem...
ERROR tests/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_cache_key_generation_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Tem...       
ERROR tests/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_timeframe_utilities_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Tem...        
ERROR tests/test_integration.py::TestOptimizationIntegration::test_end_to_end_optimization_workflow - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, g...
ERROR tests/test_integration.py::TestOptimizationIntegration::test_component_interaction_data_to_backtest - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, g...
ERROR tests/test_integration.py::TestOptimizationIntegration::test_error_scenario_data_loading_failure - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, g...
ERROR tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_success - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
ERROR tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_no_exchange - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
ERROR tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_network_error - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
ERROR tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_exchange_error - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
ERROR tests/test_live_executor.py::TestLiveOrderExecutor::test_execute_live_order_unexpected_error - TypeError: TradingSignal.__init__() missing 1 required positional argument: 'timestamp'
ERROR tests/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_context_manager - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_initialization_failure - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
ERROR tests/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_close_error_handling - TypeError: CacheConfig.__init__() got an unexpected keyword argument 'memory_config'
================= 418 failed, 2335 passed, 10 skipped, 214 warnings, 54 errors in 458.25s (0:07:38) =================
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
ValueError: I/O operation on closed file.
Call stack:
  File "C:\Users\TU\Desktop\new project\N1V1\core\metrics_endpoint.py", line 413, in _cleanup_endpoint_on_exit        
    logger.info("Performing metrics endpoint cleanup on application exit", component="metrics_endpoint", operation="cleanup")
  File "C:\Users\TU\Desktop\new project\N1V1\core\logging_utils.py", line 164, in info
    self.logger.info(sanitized_message)
Message: 'Performing metrics endpoint cleanup on application exit | component=metrics_endpoint | operation=cleanup'   
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
ValueError: I/O operation on closed file.
Call stack:
  File "C:\Users\TU\Desktop\new project\N1V1\core\metrics_endpoint.py", line 416, in _cleanup_endpoint_on_exit
    asyncio.run(stop_metrics_endpoint())
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py", line 39, in run
    loop = events.new_event_loop()
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\asyncio\events.py", line 783, in new_event_loop       
    return get_event_loop_policy().new_event_loop()
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\asyncio\events.py", line 673, in new_event_loop       
    return self._loop_factory()
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\asyncio\windows_events.py", line 315, in __init__     
    super().__init__(proactor)
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\asyncio\proactor_events.py", line 630, in __init__    
    logger.debug('Using proactor: %s', proactor.__class__.__name__)
Message: 'Using proactor: %s'
Arguments: ('IocpProactor',)
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
ValueError: I/O operation on closed file.
Call stack:
  File "C:\Users\TU\Desktop\new project\N1V1\core\metrics_endpoint.py", line 417, in _cleanup_endpoint_on_exit        
    logger.info("Metrics endpoint cleanup completed successfully", component="metrics_endpoint", operation="cleanup") 
  File "C:\Users\TU\Desktop\new project\N1V1\core\logging_utils.py", line 164, in info
    self.logger.info(sanitized_message)
Message: 'Metrics endpoint cleanup completed successfully | component=metrics_endpoint | operation=cleanup'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
ValueError: I/O operation on closed file.
Call stack:
  File "C:\Users\TU\Desktop\new project\N1V1\core\cache.py", line 825, in _cleanup_cache_on_exit
    logger.info("Performing cache cleanup on application exit", component="cache", operation="cleanup")
  File "C:\Users\TU\Desktop\new project\N1V1\core\logging_utils.py", line 164, in info
    self.logger.info(sanitized_message)
Message: 'Performing cache cleanup on application exit | component=cache | operation=cleanup'
Arguments: ()
--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\logging\__init__.py", line 1103, in emit
    stream.write(msg + self.terminator)
ValueError: I/O operation on closed file.
Call stack:
  File "C:\Users\TU\Desktop\new project\N1V1\core\cache.py", line 828, in _cleanup_cache_on_exit
    logger.info("Cache cleanup completed successfully", component="cache", operation="cleanup")
  File "C:\Users\TU\Desktop\new project\N1V1\core\logging_utils.py", line 164, in info
    self.logger.info(sanitized_message)
Message: 'Cache cleanup completed successfully | component=cache | operation=cleanup'
Arguments: ()