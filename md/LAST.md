FAILED tests/backtest/test_backtester.py::TestRegimeAwareMetrics::test_regime_aware_with_mismatched_lengths - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 2 records...
FAILED tests/backtest/test_backtester.py::TestRegimeAwareMetrics::test_regime_aware_with_insufficient_data - AssertionError: assert 'per_regime_metrics' in {'overall': {'equity_curve': [100.0], 'losses': 0, 'max_drawdown': 0...
FAILED tests/backtest/test_backtester.py::TestPerformanceOptimizations::test_vectorized_returns_calculation - NameError: name 'isfinite' is not defined
FAILED tests/backtest/test_backtester.py::TestPerformanceOptimizations::test_pandas_regime_grouping_performance - AssertionError: assert 'per_regime_metrics' in {'overall': {'equity_curve': [1000.0, 1001.0, 1002.0, 1003.0, 1004.0...
FAILED tests/backtest/test_backtester.py::TestValidationFunctions::test_validate_equity_progression_with_invalid_types - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestValidationError'>
FAILED tests/backtest/test_backtester.py::TestRegimeAwareFunctions::test_compute_regime_aware_metrics_with_no_regime_data - KeyError: 'total_regimes'
FAILED tests/backtest/test_backtester.py::TestRegimeAwareFunctions::test_align_regime_data_lengths_with_mismatch - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 2 records...
FAILED tests/backtest/test_backtester.py::TestHelperFunctions::test_export_regime_csv_summary - AssertionError: assert 'bull' in 'Regime,Total Return,Sharpe Ratio,Win Rate,Max Drawdown,Total Trades,Avg Confidenc...
FAILED tests/backtest/test_backtester.py::TestErrorHandling::test_export_equity_progression_with_io_error - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestSecurityError'>
FAILED tests/backtest/test_backtester.py::TestErrorHandling::test_compute_backtest_metrics_with_invalid_data - numpy.core._exceptions._UFuncNoLoopError: ufunc 'maximum' did not contain a loop with signature matching types (dty...
FAILED tests/backtest/test_backtester.py::TestFileOperations::test_export_with_permission_denied - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestSecurityError'>
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_no_regime_data - KeyError: 'total_return'
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_with_regime_data - KeyError: 'total_trades'
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_missing_regime_data - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 3 records...
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareExport::test_export_regime_aware_report - KeyError: 'report_type'
FAILED tests/backtest/test_regime_aware_backtester.py::TestEdgeCases::test_regime_data_longer_than_equity - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 1 records...
FAILED tests/backtest/test_regime_aware_backtester.py::TestEdgeCases::test_regime_data_with_errors - KeyError: 'trade_count'
FAILED tests/backtest/test_regime_aware_backtester.py::TestEdgeCases::test_insufficient_data_regime - KeyError: 'avg_confidence'
FAILED tests/data/test_data.py::test_cache_operations - AssertionError: assert False
FAILED tests/data/test_data_fetcher.py::TestDataFetcherInitialization::test_init_with_config - AssertionError: assert 'C:\\Users\\T...\\.test_cache' == '.test_cache'
FAILED tests/data/test_data_fetcher.py::TestDataFetcherInitialization::test_init_cache_directory_creation - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_with_caching - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
FAILED tests/data/test_data_fetcher.py::TestDataFetcherErrorScenarios::test_cache_operations_error_handling - data.data_fetcher.PathTraversalError: Invalid cache directory path: /invalid/path
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_critical_cache_raises_exception_on_failure - AssertionError: CacheLoadError not raised
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_all_strategies_fail - AssertionError: <coroutine object DataFetcher._load_from_cache at 0x000001AF2CBA1A80> is not None
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_1_integer_ms - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_2_datetime_column - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_3_format_parsing - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timezone_normalization - TypeError: Object of type Timestamp is not JSON serializable
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_forward_fill_strategy_with_gaps - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_interpolation_strategy - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_reject_strategy_with_gaps - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_unknown_gap_strategy_defaults_to_forward_fill - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestDataOptimizations::test_cache_save_optimization - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestDataOptimizations::test_concatenation_generator_optimization - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestRefactoredFunctions::test_convert_to_dataframe - AssertionError: Lists differ: ['open', 'high', 'low', 'close', 'volume'] != ['timestamp', 'open', 'high', 'low', 'c...
FAILED tests/data/test_data_fixes.py::TestRefactoredFunctions::test_exchange_wrapper_properties - AttributeError: 'DataFetcher' object has no attribute '_ExchangeWrapper'
FAILED tests/data/test_data_fixes.py::TestStructuredLogging::test_data_fetcher_structured_logging - AssertionError: Expected 'info' to have been called.
FAILED tests/data/test_data_fixes.py::TestStructuredLogging::test_historical_loader_structured_logging - AssertionError: Expected 'info' to have been called.
FAILED tests/data/test_data_fixes.py::TestDatasetVersioningStructuredLogging::test_create_version_structured_logging - NameError: name 'time' is not defined
FAILED tests/data/test_data_fixes.py::TestDatasetVersioningStructuredLogging::test_migrate_legacy_dataset_structured_logging - AttributeError: <module 'data.dataset_versioning' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\data\\dataset_ve...
FAILED tests/data/test_data_module_refactoring.py::TestPagination::test_execute_pagination_loop_with_data - assert 2 == 1
FAILED tests/data/test_data_module_refactoring.py::TestPagination::test_infinite_loop_detection - assert True is False
FAILED tests/data/test_data_module_refactoring.py::TestPathTraversal::test_sanitize_cache_path_valid - KeyError: 'name'
FAILED tests/data/test_data_module_refactoring.py::TestPathTraversal::test_sanitize_cache_path_traversal - KeyError: 'name'
FAILED tests/data/test_data_module_refactoring.py::TestBackwardCompatibility::test_all_public_methods_still_exist - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_valid_relative - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_traversal_dots - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_absolute_path - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_backslash_traversal - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_complex_traversal - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_valid_nested - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_empty_string - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_cache_disabled_no_validation - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestDatasetVersionManagerSecurity::test_create_version_with_validation - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestDatasetVersionManagerSecurity::test_create_version_with_invalid_dataframe - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestDatasetVersionManagerSecurity::test_migrate_legacy_dataset_validation - AttributeError: <module 'data.dataset_versioning' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\data\\dataset_ve...
FAILED tests/data/test_data_security.py::TestDatasetVersionManagerSecurity::test_migrate_legacy_dataset_invalid_input - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestVersionNameSanitization::test_sanitize_version_name_absolute_paths - AssertionError: Regex pattern did not match.
FAILED tests/data/test_data_security.py::TestVersionNameSanitization::test_sanitize_version_name_invalid_characters - AssertionError: Regex pattern did not match.
FAILED tests/data/test_data_security.py::TestVersionNameSanitization::test_create_version_with_malicious_name - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestVersionNameSanitization::test_create_version_with_valid_name - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestConfigurationValidation::test_validate_data_directory_absolute_paths - AssertionError: Regex pattern did not match.
FAILED tests/data/test_data_security.py::TestConfigurationValidation::test_validate_data_directory_invalid_characters - AssertionError: Regex pattern did not match.
FAILED tests/data/test_data_security.py::TestConfigurationValidation::test_setup_data_directory_with_valid_config - AssertionError: assert False
FAILED tests/data/test_data_security.py::TestIntegrationSecurity::test_full_data_pipeline_security - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestIntegrationSecurity::test_error_handling_and_logging - KeyError: 'name'
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderInitialization::test_init_with_config - AssertionError: assert 'C:\\Users\\T...storical_data' == 'test_historical_data'
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderInitialization::test_init_default_values - AssertionError: assert 'C:\\Users\\T...storical_data' == 'historical_data'
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderInitialization::test_setup_data_directory - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\...
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_historical_data_success - NameError: name 'time' is not defined
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_historical_data_partial_failure - NameError: name 'time' is not defined
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_symbol_data_from_cache - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\...
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_symbol_data_force_refresh - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\...
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_fallback - AssertionError: assert False
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_adaptive_pricing_applied - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_small_limit_order_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_large_order_twap_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_high_spread_dca_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/integration/test_distributed_system.py::TestDistributedTaskManager::test_task_failure_retry - assert 1 == 2
FAILED tests/integration/test_distributed_system.py::TestQueueAdapters::test_rabbitmq_adapter_mock - AttributeError: <module 'core.task_manager' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\task_manager.py'...
FAILED tests/integration/test_distributed_system.py::TestQueueAdapters::test_kafka_adapter_mock - AttributeError: <module 'core.task_manager' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\task_manager.py'...
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_signal_scheduling - TypeError: TaskManager.enqueue_signal_task() got an unexpected keyword argument 'correlation_id'
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_backtest_scheduling - TypeError: TaskManager.enqueue_backtest_task() got an unexpected keyword argument 'correlation_id'
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_optimization_scheduling - TypeError: TaskManager.enqueue_optimization_task() got an unexpected keyword argument 'correlation_id'
FAILED tests/integration/test_distributed_system.py::TestFaultTolerance::test_worker_crash_recovery - assert 1 == 2
FAILED tests/integration/test_distributed_system.py::TestFaultTolerance::test_queue_adapter_failure_recovery - RuntimeError: Failed to enqueue task b50319bf-eae7-4df7-bc8b-177ba550c877
FAILED tests/integration/test_ml_serving_integration.py::TestMLServingIntegration::test_single_prediction_integration - assert 500 == 200
FAILED tests/integration/test_ml_serving_integration.py::TestMLServingIntegration::test_batch_prediction_integration - assert 500 == 200
FAILED tests/integration/test_ml_serving_integration.py::TestConcurrentLoad::test_concurrent_predictions - AssertionError: Expected 10 successful requests, got 0
FAILED tests/integration/test_ml_serving_integration.py::TestAccuracyValidation::test_prediction_consistency - assert 500 == 200
FAILED tests/integration/test_ml_serving_integration.py::TestPerformanceBenchmarks::test_latency_under_load - assert 500 == 200
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_complete_order_flow_paper_trading - assert None is not None
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_multiple_orders - assert False
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_portfolio_mode - assert None is not None
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_live_mode_simulation - TypeError: getattr(): attribute name must be string
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_balance_and_equity_calculation - AssertionError: assert Decimal('0') == Decimal('1000.0')
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_concurrent_order_execution - assert 0 == 10
FAILED tests/knowledge_base/test_knowledge_base.py::TestStorageBackends::test_json_storage - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpa4i0e7rn\test_knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestStorageBackends::test_csv_storage - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmp3l9f8_p0\test_knowledge.csv
FAILED tests/knowledge_base/test_knowledge_base.py::TestStorageBackends::test_sqlite_storage - AssertionError: assert None
FAILED tests/knowledge_base/test_knowledge_base.py::TestAdaptiveWeighting::test_market_similarity_calculation - AttributeError: 'AdaptiveWeightingEngine' object has no attribute '_calculate_market_similarity'
FAILED tests/knowledge_base/test_knowledge_base.py::TestAdaptiveWeighting::test_performance_score_calculation - AttributeError: 'AdaptiveWeightingEngine' object has no attribute '_calculate_performance_score'
FAILED tests/knowledge_base/test_knowledge_base.py::TestAdaptiveWeighting::test_update_knowledge_from_trade - assert None
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_knowledge_manager_creation - AttributeError: 'KnowledgeManager' object has no attribute 'storage'
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_store_trade_knowledge - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpfxblfcjx\knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_get_adaptive_weights - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpj6de1zsv\knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_knowledge_statistics - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmphtk2mtik\knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_disabled_knowledge_manager - AttributeError: 'KnowledgeManager' object has no attribute 'storage'
FAILED tests/knowledge_base/test_knowledge_base.py::TestWeightingCalculator::test_calculate_strategy_weight_with_knowledge - TypeError: unhashable type: 'MarketCondition'
FAILED tests/knowledge_base/test_knowledge_base.py::TestCacheManager::test_cache_manager_initialization - assert <knowledge_base.adaptive.LRUCache object at 0x000001AF2CBDEF20> == {}
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeValidator::test_validate_query_parameters_valid - TypeError: KnowledgeQuery.__init__() got an unexpected keyword argument 'max_confidence'
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeValidator::test_validate_query_parameters_invalid - TypeError: KnowledgeQuery.__init__() got an unexpected keyword argument 'max_confidence'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_save_entry - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_get_entry - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_query_entries - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_list_entries - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_get_stats - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_clear_all_success - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_clear_all_with_entries - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestEdgeCases::test_corrupted_storage_file - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmplin81agn\corrupted.json
FAILED tests/ml/test_features.py::TestCrossAssetFeatureGenerator::test_generate_spread_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestCrossAssetFeatureGenerator::test_generate_ratio_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestCrossAssetFeatureGenerator::test_missing_price_columns_error - AssertionError: Regex pattern did not match.
FAILED tests/ml/test_features.py::TestTimeAnchoredFeatureGenerator::test_generate_momentum_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestTimeAnchoredFeatureGenerator::test_generate_volume_profile_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestFeatureDriftDetector::test_kolmogorov_smirnov_drift_detection - assert 0.053 < 0.05
FAILED tests/ml/test_features.py::TestFeatureDriftDetector::test_population_stability_index_drift_detection - assert 0.048607825577174996 > 0.1
FAILED tests/ml/test_features.py::TestFeatureDriftDetector::test_missing_features_error - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/ml/test_features.py::TestFeatureValidation::test_validate_feature_importance_stability - AttributeError: <module 'ml.features' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\features.py'> does not h...
FAILED tests/ml/test_features.py::TestIntegrationFeatures::test_full_feature_pipeline - ValueError: Data must contain 'close' column
FAILED tests/ml/test_ml.py::TestTrainFunctions::test_load_historical_data_csv - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/ml/test_ml.py::TestTrainFunctions::test_prepare_training_data - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'op...
FAILED tests/ml/test_ml.py::TestTrainFunctions::test_prepare_training_data_insufficient_samples - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'op...
FAILED tests/ml/test_ml.py::TestTrainerFunctions::test_validate_inputs - TypeError: validate_inputs() missing 1 required positional argument: 'feature_columns'
FAILED tests/ml/test_ml.py::TestTrainerFunctions::test_validate_inputs_missing_label - TypeError: validate_inputs() missing 1 required positional argument: 'feature_columns'
FAILED tests/ml/test_ml.py::TestModelLoaderFunctions::test_predict - TypeError: 'Mock' object is not iterable
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_all_indicators - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_bollinger_bands - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_ema - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_macd - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_rsi - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_insufficient_data_rsi - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_validate_ohlcv_data - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestIntegration::test_full_pipeline - ValueError: Training data cannot be empty
ERROR tests/data/test_data.py::test_historical_loader - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data.py::test_historical_data_validation - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data.py::test_data_resampling - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data.py::test_historical_data_pagination - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_get_cache_key - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_empty - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_and_load_from_cache - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_to_cache_empty_dataframe - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_expired - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_full_workflow_with_caching - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_multiple_symbols_workflow - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_realtime_data_workflow - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local...
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_save_to_cache_success - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_load_from_cache_success - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_load_from_cache_nonexistent - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_load_from_cache_expired - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_cache_key_generation - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestVersioning::test_create_version_success - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/data/test_data_module_refactoring.py::TestVersioning::test_create_version_invalid_name - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/data/test_data_module_refactoring.py::TestVersioning::test_migrate_legacy_dataset - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_full_data_loading_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\...
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_data_validation_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\...
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_cache_key_generation_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\...
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_timeframe_utilities_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! KeyboardInterrupt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py:320: KeyboardInterrupt
(to show a full traceback on KeyboardInterrupt use --full-trace)
======================== 145 failed, 430 passed, 102 warnings, 24 errors in 1384.94s (0:23:04) ========================