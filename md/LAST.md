================================================ short test summary info ================================================
FAILED tests/acceptance/test_docs.py::TestDocumentationValidation::test_quickstart_guide_exists_and_is_complete - AssertionError: Required section 'Installation' missing from Quickstart guide
FAILED tests/acceptance/test_docs.py::TestDocumentationValidation::test_setup_instructions_work_correctly - AssertionError: pytest not found in requirements.txt
FAILED tests/acceptance/test_docs.py::TestDocumentationValidation::test_troubleshooting_section_completeness - AssertionError: Troubleshooting section only covers 0 out of 5 common issues
FAILED tests/acceptance/test_docs.py::TestDocumentationValidation::test_documentation_accessibility_validation - AssertionError: Documentation file README.md is too small (76 bytes)
FAILED tests/acceptance/test_ml_quality.py::TestMLQualityValidation::test_regression_detection_over_time - assert False == True
FAILED tests/acceptance/test_scalability.py::TestScalabilityValidation::test_multi_node_scaling - AttributeError: 'TaskManager' object has no attribute 'register_worker'
FAILED tests/acceptance/test_scalability.py::TestScalabilityValidation::test_no_downtime_during_scaling - AttributeError: 
'TaskManager' object has no attribute 'register_worker'
FAILED tests/acceptance/test_scalability.py::TestScalabilityValidation::test_distributed_task_processing - AttributeError: 'TaskManager' object has no attribute 'register_worker'
FAILED tests/acceptance/test_scalability.py::TestScalabilityValidation::test_cluster_failure_recovery - AttributeError: 'TaskManager' object has no attribute 'register_worker'
FAILED tests/acceptance/test_slo.py::TestSLOValidation::test_system_throughput_under_load - AttributeError: <core.signal_processor.SignalProcessor object at 0x000002891635B7C0> does not have the attribute 'pro...
FAILED tests/acceptance/test_slo.py::TestSLOValidation::test_end_to_end_latency_slo - AttributeError: <core.signal_processor.SignalProcessor object at 0x0000028919F85C00> does not have the attribute 'pro...
FAILED tests/acceptance/test_slo.py::TestSLOValidation::test_slo_validation_report_generation - KeyError: 'all_slo_targets_met'
FAILED tests/acceptance/test_stability.py::TestStabilityValidation::test_exchange_failure_rollback - AttributeError: 'NoneType' object has no attribute 'execute_live_order'
FAILED tests/acceptance/test_stability.py::TestStabilityValidation::test_failover_recovery_mechanism - assert True == False
FAILED tests/backtest/test_backtester.py::TestRegimeAwareMetrics::test_regime_aware_with_mismatched_lengths - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 2 records, ...
FAILED tests/backtest/test_backtester.py::TestRegimeAwareMetrics::test_regime_aware_with_insufficient_data - AssertionError: assert 'per_regime_metrics' in {'overall': {'equity_curve': [100.0], 'losses': 0, 'max_drawdown': 0.0...
FAILED tests/backtest/test_backtester.py::TestPerformanceOptimizations::test_vectorized_returns_calculation - NameError: name 'isfinite' is not defined
FAILED tests/backtest/test_backtester.py::TestPerformanceOptimizations::test_pandas_regime_grouping_performance - AssertionError: assert 'per_regime_metrics' in {'overall': {'equity_curve': [1000.0, 1001.0, 1002.0, 1003.0, 1004.0, ...
FAILED tests/backtest/test_backtester.py::TestValidationFunctions::test_validate_equity_progression_with_invalid_types - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestValidationError'>
FAILED tests/backtest/test_backtester.py::TestRegimeAwareFunctions::test_compute_regime_aware_metrics_with_no_regime_data 
- KeyError: 'total_regimes'
FAILED tests/backtest/test_backtester.py::TestRegimeAwareFunctions::test_align_regime_data_lengths_with_mismatch - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 2 records, ...
FAILED tests/backtest/test_backtester.py::TestHelperFunctions::test_export_regime_csv_summary - AssertionError: assert 'bull' in 'Regime,Total Return,Sharpe Ratio,Win Rate,Max Drawdown,Total Trades,Avg Confidence\...
FAILED tests/backtest/test_backtester.py::TestErrorHandling::test_export_equity_progression_with_io_error - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestSecurityError'>
FAILED tests/backtest/test_backtester.py::TestErrorHandling::test_compute_backtest_metrics_with_invalid_data - numpy.core._exceptions._UFuncNoLoopError: ufunc 'maximum' did not contain a loop with signature matching types (dtype...
FAILED tests/backtest/test_backtester.py::TestFileOperations::test_export_with_permission_denied - Failed: DID NOT RAISE <class 'backtest.backtester.BacktestSecurityError'>
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_no_regime_data - KeyError: 'total_return'
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_with_regime_data - KeyError: 'total_trades'
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareMetrics::test_compute_regime_aware_metrics_missing_regime_data - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 3 records, ...
FAILED tests/backtest/test_regime_aware_backtester.py::TestRegimeAwareExport::test_export_regime_aware_report - KeyError: 
'report_type'
FAILED tests/backtest/test_regime_aware_backtester.py::TestEdgeCases::test_regime_data_longer_than_equity - backtest.backtester.BacktestValidationError: Critical regime data length mismatch: equity_progression has 1 records, ...
FAILED tests/backtest/test_regime_aware_backtester.py::TestEdgeCases::test_regime_data_with_errors - KeyError: 'trade_count'
FAILED tests/backtest/test_regime_aware_backtester.py::TestEdgeCases::test_insufficient_data_regime - KeyError: 'avg_confidence'
FAILED tests/data/test_data.py::test_cache_operations - AssertionError: assert False
FAILED tests/data/test_data_fetcher.py::TestDataFetcherInitialization::test_init_with_config - AssertionError: assert 'C:\\Users\\T...\\.test_cache' == '.test_cache'
FAILED tests/data/test_data_fetcher.py::TestDataFetcherInitialization::test_init_cache_directory_creation - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_with_caching - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
FAILED tests/data/test_data_fetcher.py::TestDataFetcherErrorScenarios::test_cache_operations_error_handling - data.data_fetcher.PathTraversalError: Invalid cache directory path: /invalid/path
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_critical_cache_raises_exception_on_failure - 
AssertionError: CacheLoadError not raised
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_all_strategies_fail - AssertionError: <coroutine object DataFetcher._load_from_cache at 0x000002891A15C9E0> is not None
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_1_integer_ms - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_2_datetime_column 
- AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_3_format_parsing - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timezone_normalization - TypeError: Object of type Timestamp is not JSON serializable
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_forward_fill_strategy_with_gaps - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_interpolation_strategy - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_reject_strategy_with_gaps - NameError: name 'np' is 
not defined
FAILED tests/data/test_data_fixes.py::TestGapHandlingStrategies::test_unknown_gap_strategy_defaults_to_forward_fill - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestDataOptimizations::test_cache_save_optimization - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestDataOptimizations::test_concatenation_generator_optimization - NameError: name 'np' is not defined
FAILED tests/data/test_data_fixes.py::TestRefactoredFunctions::test_convert_to_dataframe - AssertionError: Lists differ: ['open', 'high', 'low', 'close', 'volume'] != ['timestamp', 'open', 'high', 'low', 'clo...
FAILED tests/data/test_data_fixes.py::TestRefactoredFunctions::test_exchange_wrapper_properties - AttributeError: 'DataFetcher' object has no attribute '_ExchangeWrapper'
FAILED tests/data/test_data_fixes.py::TestStructuredLogging::test_data_fetcher_structured_logging - AssertionError: Expected 'info' to have been called.
FAILED tests/data/test_data_fixes.py::TestStructuredLogging::test_historical_loader_structured_logging - AssertionError: Expected 'info' to have been called.
FAILED tests/data/test_data_fixes.py::TestDatasetVersioningStructuredLogging::test_create_version_structured_logging - NameError: name 'time' is not defined
FAILED tests/data/test_data_fixes.py::TestDatasetVersioningStructuredLogging::test_migrate_legacy_dataset_structured_logging - AttributeError: <module 'data.dataset_versioning' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\data\\dataset_vers...
FAILED tests/data/test_data_module_refactoring.py::TestPagination::test_execute_pagination_loop_with_data - assert 2 == 1 
FAILED tests/data/test_data_module_refactoring.py::TestPagination::test_infinite_loop_detection - assert True is False    
FAILED tests/data/test_data_module_refactoring.py::TestPathTraversal::test_sanitize_cache_path_valid - KeyError: 'name'   
FAILED tests/data/test_data_module_refactoring.py::TestPathTraversal::test_sanitize_cache_path_traversal - KeyError: 'name'
FAILED tests/data/test_data_module_refactoring.py::TestBackwardCompatibility::test_all_public_methods_still_exist - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_valid_relative - KeyError: 
'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_traversal_dots - KeyError: 
'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_absolute_path - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_backslash_traversal - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_complex_traversal - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_valid_nested - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_sanitize_cache_path_empty_string - KeyError: 'name'
FAILED tests/data/test_data_security.py::TestPathTraversalPrevention::test_cache_disabled_no_validation - KeyError: 'name'FAILED tests/data/test_data_security.py::TestDatasetVersionManagerSecurity::test_create_version_with_validation - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestDatasetVersionManagerSecurity::test_create_version_with_invalid_dataframe - NameError: name 'time' is not defined
FAILED tests/data/test_data_security.py::TestDatasetVersionManagerSecurity::test_migrate_legacy_dataset_validation - AttributeError: <module 'data.dataset_versioning' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\data\\dataset_vers...       
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
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderInitialization::test_setup_data_directory - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\tm...
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_historical_data_success - NameError: name 'time' is not defined
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_historical_data_partial_failure - NameError: name 'time' is not defined
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_symbol_data_from_cache - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\tm...      
FAILED tests/data/test_historical_loader.py::TestHistoricalDataLoaderAsyncMethods::test_load_symbol_data_force_refresh - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\tm...   
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_fallback - AssertionError: assert False
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_adaptive_pricing_applied - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_small_limit_order_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_large_order_twap_flow - AssertionError: assert 
<ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_high_spread_dca_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/integration/test_distributed_system.py::TestDistributedTaskManager::test_task_failure_retry - assert 1 == 2  
FAILED tests/integration/test_distributed_system.py::TestQueueAdapters::test_rabbitmq_adapter_mock - AttributeError: <module 'core.task_manager' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\task_manager.py'> ...
FAILED tests/integration/test_distributed_system.py::TestQueueAdapters::test_kafka_adapter_mock - AttributeError: <module 
'core.task_manager' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\task_manager.py'> ...
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_signal_scheduling - TypeError: TaskManager.enqueue_signal_task() got an unexpected keyword argument 'correlation_id'
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_backtest_scheduling - TypeError: TaskManager.enqueue_backtest_task() got an unexpected keyword argument 'correlation_id'
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_optimization_scheduling - TypeError: TaskManager.enqueue_optimization_task() got an unexpected keyword argument 'correlation_id'
FAILED tests/integration/test_distributed_system.py::TestFaultTolerance::test_worker_crash_recovery - assert 1 == 2       
FAILED tests/integration/test_distributed_system.py::TestFaultTolerance::test_queue_adapter_failure_recovery - RuntimeError: Failed to enqueue task c0b8e730-143e-4af1-adad-d6909ce1e321
FAILED tests/integration/test_ml_serving_integration.py::TestMLServingIntegration::test_single_prediction_integration - assert 500 == 200
FAILED tests/integration/test_ml_serving_integration.py::TestMLServingIntegration::test_batch_prediction_integration - assert 500 == 200
FAILED tests/integration/test_ml_serving_integration.py::TestConcurrentLoad::test_concurrent_predictions - AssertionError: Expected 10 successful requests, got 0
FAILED tests/integration/test_ml_serving_integration.py::TestAccuracyValidation::test_prediction_consistency - assert 500 
== 200
FAILED tests/integration/test_ml_serving_integration.py::TestPerformanceBenchmarks::test_latency_under_load - assert 500 == 200
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_complete_order_flow_paper_trading 
- assert None is not None
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_multiple_orders - assert False
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_portfolio_mode - assert None is not None
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_live_mode_simulation - 
AttributeError: 'str' object has no attribute 'get'
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_balance_and_equity_calculation - AssertionError: assert Decimal('0') == Decimal('1000.0')
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_concurrent_order_execution - assert 0 == 10
FAILED tests/knowledge_base/test_knowledge_base.py::TestStorageBackends::test_json_storage - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmp8ae9jzoi\test_knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestStorageBackends::test_csv_storage - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmplurdz48o\test_knowledge.csv
FAILED tests/knowledge_base/test_knowledge_base.py::TestStorageBackends::test_sqlite_storage - AssertionError: assert NoneFAILED tests/knowledge_base/test_knowledge_base.py::TestAdaptiveWeighting::test_market_similarity_calculation - AttributeError: 'AdaptiveWeightingEngine' object has no attribute '_calculate_market_similarity'
FAILED tests/knowledge_base/test_knowledge_base.py::TestAdaptiveWeighting::test_performance_score_calculation - AttributeError: 'AdaptiveWeightingEngine' object has no attribute '_calculate_performance_score'
FAILED tests/knowledge_base/test_knowledge_base.py::TestAdaptiveWeighting::test_update_knowledge_from_trade - assert None 
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_knowledge_manager_creation - AttributeError: 'KnowledgeManager' object has no attribute 'storage'
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_store_trade_knowledge - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpznjqg18k\knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_get_adaptive_weights - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpzbztraoc\knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_knowledge_statistics - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmpr1uemsm9\knowledge.json
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeManager::test_disabled_knowledge_manager - AttributeError: 'KnowledgeManager' object has no attribute 'storage'
FAILED tests/knowledge_base/test_knowledge_base.py::TestWeightingCalculator::test_calculate_strategy_weight_with_knowledge - TypeError: unhashable type: 'MarketCondition'
FAILED tests/knowledge_base/test_knowledge_base.py::TestCacheManager::test_cache_manager_initialization - assert <knowledge_base.adaptive.LRUCache object at 0x000002891A48AEF0> == {}
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeValidator::test_validate_query_parameters_valid - TypeError: KnowledgeQuery.__init__() got an unexpected keyword argument 'max_confidence'
FAILED tests/knowledge_base/test_knowledge_base.py::TestKnowledgeValidator::test_validate_query_parameters_invalid - TypeError: KnowledgeQuery.__init__() got an unexpected keyword argument 'max_confidence'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_save_entry - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_get_entry - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_query_entries - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_list_entries - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_get_stats - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_clear_all_success - ModuleNotFoundError: 
No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestDataStoreInterface::test_clear_all_with_entries - ModuleNotFoundError: No module named 'tests.test_knowledge_base'
FAILED tests/knowledge_base/test_knowledge_base.py::TestEdgeCases::test_corrupted_storage_file - PermissionError: Path traversal attempt detected: C:\Users\TU\AppData\Local\Temp\tmp0__87vxk\corrupted.json
FAILED tests/ml/test_features.py::TestCrossAssetFeatureGenerator::test_generate_spread_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestCrossAssetFeatureGenerator::test_generate_ratio_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestCrossAssetFeatureGenerator::test_missing_price_columns_error - AssertionError: Regex pattern did not match.
FAILED tests/ml/test_features.py::TestTimeAnchoredFeatureGenerator::test_generate_momentum_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestTimeAnchoredFeatureGenerator::test_generate_volume_profile_features - AssertionError: Series are different
FAILED tests/ml/test_features.py::TestFeatureDriftDetector::test_kolmogorov_smirnov_drift_detection - assert 0.053 < 0.05 
FAILED tests/ml/test_features.py::TestFeatureDriftDetector::test_population_stability_index_drift_detection - assert 0.048607825577174996 > 0.1
FAILED tests/ml/test_features.py::TestFeatureDriftDetector::test_missing_features_error - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/ml/test_features.py::TestFeatureValidation::test_validate_feature_importance_stability - AttributeError: <module 'ml.features' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\features.py'> does not hav...
FAILED tests/ml/test_features.py::TestIntegrationFeatures::test_full_feature_pipeline - ValueError: Data must contain 'close' column
FAILED tests/ml/test_ml.py::TestTrainFunctions::test_load_historical_data_csv - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/ml/test_ml.py::TestTrainFunctions::test_prepare_training_data - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'open...
FAILED tests/ml/test_ml.py::TestTrainFunctions::test_prepare_training_data_insufficient_samples - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'open...
FAILED tests/ml/test_ml.py::TestTrainerFunctions::test_validate_inputs - TypeError: validate_inputs() missing 1 required positional argument: 'feature_columns'
FAILED tests/ml/test_ml.py::TestTrainerFunctions::test_validate_inputs_missing_label - TypeError: validate_inputs() missing 1 required positional argument: 'feature_columns'
FAILED tests/ml/test_ml.py::TestModelLoaderFunctions::test_predict - TypeError: 'Mock' object is not iterable
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_all_indicators - AttributeError: 'TestConfiguration' object 
has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_bollinger_bands - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_ema - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_macd - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_calculate_rsi - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_insufficient_data_rsi - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestConfiguration::test_validate_ohlcv_data - AttributeError: 'TestConfiguration' object has no attribute 'test_data'
FAILED tests/ml/test_ml.py::TestIntegration::test_full_pipeline - ValueError: Training data cannot be empty
FAILED tests/ml/test_ml_filter.py::TestMLFilter::test_save_load_model - assert False
FAILED tests/ml/test_ml_filter.py::TestFactoryFunctions::test_load_ml_filter - assert False
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_python_random - xgboost.core.XGBoostError: [16:02:05] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_numpy - xgboost.core.XGBoostError: [16:02:06] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_pandas - xgboost.core.XGBoostError: [16:02:06] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_lightgbm - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_xgboost - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_catboost - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_tensorflow - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_pytorch - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_packages - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_env_vars - AssertionError: assert '/usr/local/bin:/usr/bin' == '/usr/local/lib/python3.9'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_git_info - KeyError: 'commit_hash'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_git_failure - AssertionError: assert 'error' in {}
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_hardware - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_hardware_failure - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the ...     
FAILED tests/ml/test_reproducibility.py::TestExperimentTrackerReproducibility::test_experiment_tracker_git_info_logging - 
KeyError: 'commit_hash'
FAILED tests/ml/test_reproducibility.py::TestReproducibilityValidation::test_deterministic_numpy_operations - xgboost.core.XGBoostError: [16:02:13] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityValidation::test_deterministic_pandas_operations - xgboost.core.XGBoostError: [16:02:14] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_full_reproducibility_workflow - xgboost.core.XGBoostError: [16:02:16] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_reproducibility_with_different_seeds - xgboost.core.XGBoostError: [16:02:16] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...       
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_reproducibility_with_same_seed - xgboost.core.XGBoostError: [16:02:16] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_serving.py::TestServingAPI::test_predict_single_success - assert 500 == 200
FAILED tests/ml/test_serving.py::TestPredictionProcessing::test_process_single_prediction_success - fastapi.exceptions.HTTPException
FAILED tests/ml/test_serving.py::TestPerformanceMetrics::test_metrics_collection_success - assert 500 == 200
FAILED tests/ml/test_train.py::TestLoadHistoricalData::test_load_csv_data_success - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/ml/test_train.py::TestLoadHistoricalData::test_load_data_with_symbol_filter - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/ml/test_train.py::TestLoadHistoricalData::test_load_data_missing_required_columns - AttributeError: 'str' object has no attribute 'columns'
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_success - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'open...
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_insufficient_samples - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'open...
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_with_nan_values - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'open...
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_with_zero_prices - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'open...
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_outlier_removal - KeyError: "Required column 'Open' (mapped from 'open') not found in DataFrame. Available columns: ['timestamp', 'open...
FAILED tests/ml/test_train.py::TestMainFunction::test_main_success - AssertionError: Expected 'exit' to not have been called. Called 1 times.
FAILED tests/ml/test_train.py::TestMainFunction::test_main_predictive_models_disabled - AssertionError: Expected 'exit' to not have been called. Called 1 times.
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_basic - xgboost.core.XGBoostError: [16:02:19] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_with_libs - xgboost.core.XGBoostError: [16:02:19] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_libs_unavailable - xgboost.core.XGBoostError: [16:02:20] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a...
FAILED tests/ml/test_train.py::TestReproducibility::test_capture_environment_snapshot - AssertionError: assert 'x86_64' == 'Linux-5.4.0'
FAILED tests/ml/test_train.py::TestModelLoaderReproducibility::test_load_model_from_registry_success - ModuleNotFoundError: No module named 'mlflow'
FAILED tests/ml/test_train.py::TestModelLoaderReproducibility::test_load_model_from_registry_fallback - ModuleNotFoundError: No module named 'mlflow'
FAILED tests/ml/test_trainer.py::TestDataProcessing::test_load_data_success - NameError: name 'logger' is not defined     
FAILED tests/ml/test_trainer.py::TestDataProcessing::test_load_data_file_not_found - NameError: name 'logger' is not defined
FAILED tests/optimization/test_asset_selector.py::TestAssetSelector::test_fetch_from_coingecko_async_success - AssertionError: assert None == {'ETH/USDT': 50000000000}
FAILED tests/optimization/test_cross_asset_validation.py::TestAssetSelector::test_apply_market_cap_weighting - assert 0.5 
> 0.5
FAILED tests/optimization/test_optimization.py::TestWalkForwardOptimizer::test_optimization_with_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_optimization.py::TestGeneticOptimizer::test_population_initialization - AssertionError: assert 20 == 10
FAILED tests/optimization/test_optimization.py::TestGeneticOptimizer::test_population_initialization_no_bounds - assert 20 == 5
FAILED tests/optimization/test_optimization.py::TestCoreOptimizationAlgorithms::test_crossover_with_different_gene_counts 
- KeyError: 'param2'
FAILED tests/optimization/test_optimization.py::TestCoreOptimizationAlgorithms::test_parameter_validation_edge_cases - AssertionError: assert not True
FAILED tests/optimization/test_optimization.py::TestRLOptimizer::test_policy_save_load - assert 0 > 0
FAILED tests/optimization/test_optimization.py::TestOptimizerFactory::test_create_optimizer - assert 20 == 10
FAILED tests/optimization/test_optimization.py::TestCrossPairValidation::test_cross_pair_validation_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_walk_forward.py::TestWalkForwardOptimizer::test_optimize_no_windows - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_walk_forward.py::TestWalkForwardOptimizer::test_optimize_with_windows - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/optimization/test_walk_forward.py::TestIntegration::test_full_walk_forward_workflow - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/optimization/test_walk_forward.py::TestIntegration::test_error_handling_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_walk_forward.py::TestIntegration::test_error_handling_optimizer_failure - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_normal - ValueError: 
Critical market data validation error for BTC/USDT: Market data validation failed - missing required colu...
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_high_volatility - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required colu...     
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_empty_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_none_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_market_monitor_error_handling - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required colu...
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_calculation - assert False
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_fallback - AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x000002891BB5012...
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_position_sizing_method_selection - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_error_handling_in_calculations - TypeError: '>=' not supported between instances of 'coroutine' and 'int'
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskIntegration::test_complete_trade_workflow - assert False is True
FAILED tests/risk/test_anomaly_detector.py::TestPriceGapDetector::test_detect_normal_price_change - AssertionError: assert False is False
FAILED tests/risk/test_anomaly_detector.py::TestPriceGapDetector::test_detect_price_gap - AssertionError: assert True is True
FAILED tests/risk/test_anomaly_detector.py::TestPriceGapDetector::test_detect_small_gap - AssertionError: assert False is 
False
FAILED tests/risk/test_anomaly_detector.py::TestAnomalyDetector::test_initialization_with_config - KeyError: 'scale_down_threshold'
FAILED tests/risk/test_anomaly_detector.py::TestAnomalyDetector::test_get_anomaly_statistics - AttributeError: 'AnomalyDetector' object has no attribute 'get_anomaly_statistics'
FAILED tests/risk/test_anomaly_detector.py::TestAnomalyDetector::test_empty_statistics - AttributeError: 'AnomalyDetector' object has no attribute 'get_anomaly_statistics'
FAILED tests/risk/test_anomaly_detector.py::TestAnomalyLogging::test_log_to_file - KeyError: 'enabled'
FAILED tests/risk/test_anomaly_detector.py::TestAnomalyLogging::test_log_to_json - KeyError: 'enabled'
FAILED tests/risk/test_anomaly_detector.py::TestAnomalyLogging::test_trade_logger_integration - AttributeError: 'AnomalyDetector' object has no attribute '_log_anomaly'. Did you mean: 'log_anomalies'?
FAILED tests/risk/test_anomaly_detector.py::TestEdgeCases::test_extreme_values - AssertionError: assert True is True      
FAILED tests/risk/test_anomaly_detector.py::TestConfiguration::test_custom_configuration - KeyError: 'skip_trade_threshold'
FAILED tests/risk/test_anomaly_detector.py::TestConfiguration::test_severity_threshold_conversion - KeyError: 'scale_down_factor'
FAILED tests/risk/test_risk.py::TestAnomalyDetection::test_price_zscore_detector_anomalous_data - assert False
FAILED tests/risk/test_risk.py::TestAnomalyDetection::test_volume_zscore_detector_anomalous_volume - assert 0 > 0
FAILED tests/risk/test_risk.py::TestIntegrationTests::test_full_risk_assessment_workflow - assert False is True
FAILED tests/risk/test_risk.py::TestExtremeConditions::test_flash_crash_simulation - AssertionError: assert <AnomalyType.PRICE_ZSCORE: 'price_zscore'> in {<AnomalyType.PRICE_GAP: 'price_gap'>}
FAILED tests/risk/test_risk.py::TestExtremeConditions::test_extreme_volume_spike - assert 0 > 0
FAILED tests/risk/test_risk.py::TestRiskManagerEdgeCases::test_risk_manager_zero_balance - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk.py::TestRiskManagerEdgeCases::test_risk_manager_extreme_price_values - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_position_sizing_with_adaptive_policy - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_policy_called_with_correct_data - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_kill_switch_blocks_trading - 
decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_defensive_mode_reduces_position_size - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_atr_position_sizing_with_multiplier - AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x000002891B3CB76...
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_error_handling_in_adaptive_integration - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_different_position_sizing_methods_with_adaptive - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_multiplier_application - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerPerformanceIntegration::test_trade_outcome_updates_adaptive_policy - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerPerformanceIntegration::test_consecutive_losses_affect_position_sizing - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/security/test_key_management.py::TestAWSKMSKeyManager::test_get_secret_success - ModuleNotFoundError: No module named 'boto3'
FAILED tests/security/test_key_management.py::TestAWSKMSKeyManager::test_store_secret_success - ModuleNotFoundError: No module named 'boto3'
FAILED tests/security/test_key_management.py::TestSecureCredentialManager::test_rotate_key_local - assert False is True   
FAILED tests/security/test_secret_manager.py::TestSecretManager::test_get_secret_vault_success - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/security/test_secret_manager.py::TestSecretManager::test_rotate_key_success - assert False is True
FAILED tests/security/test_secret_manager.py::TestAWSKMSKeyManager::test_get_secret_success - ModuleNotFoundError: No module named 'boto3'
FAILED tests/security/test_secret_manager.py::TestAWSKMSKeyManager::test_store_secret_success - ModuleNotFoundError: No module named 'boto3'
FAILED tests/security/test_secret_manager.py::TestAWSKMSKeyManager::test_health_check_success - ModuleNotFoundError: No module named 'boto3'
FAILED tests/strategies/test_strategies.py::TestBollingerReversionStrategy::test_generate_signals_oversold_reversion - assert 0 == 1
FAILED tests/strategies/test_strategies.py::TestBollingerReversionStrategy::test_generate_signals_overbought_reversion - assert 0 == 1
FAILED tests/strategies/test_strategies.py::TestMockingAndIntegration::test_signal_generation_deterministic_timestamps - assert 0 == 1
FAILED tests/strategies/test_strategy_generator.py::TestStrategyGenerator::test_strategy_generation - assert None is not None
FAILED tests/strategies/test_strategy_generator.py::TestPerformance::test_generation_performance - assert None is not NoneFAILED tests/strategies/test_strategy_integration.py::TestStrategyIntegrationWorkflow::test_strategy_error_handling_workflow - assert 0 >= 2
FAILED tests/strategies/test_strategy_integration.py::TestStrategyRiskManagerIntegration::test_risk_manager_position_sizing_integration - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/strategies/test_strategy_integration.py::TestStrategyDataFetcherIntegration::test_data_fetcher_error_recovery_integration - assert 1 == 2
FAILED tests/test_integration.py::TestOptimizationIntegration::test_output_validation_backtest_metrics - AssertionError: assert False
FAILED tests/test_integration.py::TestOptimizationIntegration::test_error_scenario_optimization_with_invalid_parameters - 
AttributeError: 'GeneticOptimizer' object has no attribute 'ParameterBounds'. Did you mean: 'parameter_bounds'?
FAILED tests/unit/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_baseline_coefficient_variation_edge_cases - assert False
FAILED tests/unit/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_anomaly_detection_zero_std - TypeError: object of type 'coroutine' has no len()
FAILED tests/unit/test_algorithmic_correctness.py::TestPerformanceMonitorCorrectness::test_percentile_anomaly_score_calculation - TypeError: object of type 'coroutine' has no len()
FAILED tests/unit/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_decimal_initial_balance_conversion - AssertionError: assert False
FAILED tests/unit/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_invalid_initial_balance_fallback - AssertionError: assert False
FAILED tests/unit/test_algorithmic_correctness.py::TestOrderManagerPrecision::test_none_initial_balance_fallback - AssertionError: assert False
FAILED tests/unit/test_algorithmic_correctness.py::TestStatisticalEdgeCases::test_percentile_calculation_robustness - assert 4.8 >= 4.96
FAILED tests/unit/test_api_app.py::TestCustomExceptionMiddleware::test_normal_request_passthrough - AssertionError: Expected 'mock' to have been called once. Called 0 times.
FAILED tests/unit/test_api_app.py::TestAPIEndpoints::test_health_endpoint_no_bot_engine - assert 500 == 200
FAILED tests/unit/test_binary_integration.py::TestBinaryModelIntegrationEdgeCases::test_binary_model_prediction_with_exact_threshold - assert False == True
FAILED tests/unit/test_binary_integration.py::TestBinaryModelIntegrationEdgeCases::test_binary_model_prediction_with_low_confidence - assert -1.0 == 0.55
FAILED tests/unit/test_binary_integration.py::TestGlobalIntegrationFunctions::test_integrate_binary_model_convenience_function - NameError: name 'integrate_binary_model' is not defined
FAILED tests/unit/test_binary_integration.py::TestErrorRecovery::test_binary_model_prediction_error_recovery - assert -1.0 == 0.0
FAILED tests/unit/test_binary_integration.py::TestErrorRecovery::test_strategy_selection_error_recovery - AssertionError: 
assert <MarketRegime.UNKNOWN: 'unknown'> == 'UNKNOWN'
FAILED tests/unit/test_binary_integration.py::TestErrorRecovery::test_complete_pipeline_error_recovery - AssertionError: assert 'Integration error' in 'Binary integration disabled'
FAILED tests/unit/test_binary_integration.py::TestPerformanceAndScalability::test_concurrent_market_data_processing - AssertionError: assert False == True
FAILED tests/unit/test_binary_integration.py::TestLoggingAndMonitoring::test_binary_prediction_logging - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationFeatureExtraction::test_calculate_macd_edge_cases - assert 0.02243589743588359 == 0.0
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationGlobalFunctions::test_get_binary_integration_with_config - assert False == True
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationMetricsIntegration::test_metrics_recording_in_prediction - AssertionError: Expected 'record_prediction' to have been called once. Called 0 times.
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_predict_binary_model_with_model_exception - assert -1.0 == 0.0
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_select_strategy_with_regime_detector_failure - AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == <MarketRegime.UNKNOWN: 'unknown'>   
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationLogging::test_binary_prediction_logging_integration - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/unit/test_binary_model_metrics.py::TestMetricsCollection::test_collect_binary_model_metrics_with_exception - 
AssertionError: assert False
FAILED tests/unit/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_trade_frequency_drift - AssertionError: assert 'trade frequency changed' in 'Trade frequency changed by 100.0%'
FAILED tests/unit/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_accuracy_drop - assert 0 == 1  
FAILED tests/unit/test_binary_model_metrics.py::TestPerformanceReporting::test_get_performance_report - KeyError: 'was_correct'
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_collect_metrics_with_binary_integration_import_error - AttributeError: <module 'core.binary_model_metrics' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\binary_mod...
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_record_prediction_with_invalid_data - TypeError: '>=' not supported between instances of 'str' and 'float'
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_record_decision_outcome_with_invalid_data - TypeError: unsupported operand type(s) for +=: 'float' and 'str'
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_calculate_metrics_with_corrupted_data - TypeError: '>' not supported between instances of 'str' and 'datetime.datetime'
FAILED tests/unit/test_binary_model_metrics.py::TestDataIntegrity::test_prediction_history_data_integrity - TypeError: BinaryModelMetricsCollector.record_prediction() got an unexpected keyword argument 'decision'
FAILED tests/unit/test_bot_engine.py::TestBotEngine::test_trading_cycle - AssertionError: Expected 'get_historical_data' to have been called once. Called 0 times.
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_minimal_config - AssertionError: assert [] == ['BTC/USDT']
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_portfolio_mode - AssertionError: assert set() == {'BTC/USDT', 'ETH/USDT'}
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_invalid_balance 
- assert 10000.0 == 1000.0
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_empty_markets - 
AssertionError: assert [] == ['BTC/USDT']
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_calculate_max_drawdown - assert 0.061224489795918366 == 0.08
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_success - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_missing_equity - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineStateManagement::test_run_main_loop_normal_operation - AssertionError: Expected '_update_display' to have been called once. Called 0 times.
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_process_binary_integration_enabled - assert 0 == 1
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_execute_integrated_decisions - AssertionError: Expected 'execute_order' to have been called once. Called 0 times.
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_trading_cycle_with_data_fetch_error - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_signal_generation_with_strategy_error - Exception: Strategy failed
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_risk_evaluation_with_component_error 
- AttributeError: 'NoneType' object has no attribute 'evaluate_signal'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_order_execution_with_component_error 
- AttributeError: 'NoneType' object has no attribute 'execute_order'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_state_update_with_component_errors - 
AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_empty_market_data - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_none_market_data - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_concurrent_trading_cycles - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_cache_comprehensive.py::TestCacheBatchOperations::test_set_multiple_ohlcv_success - assert False == True
FAILED tests/unit/test_cache_comprehensive.py::TestCacheInvalidation::test_invalidate_symbol_data_specific_types - assert 
3 == 1
FAILED tests/unit/test_cache_comprehensive.py::TestCacheContextManager::test_context_manager_success - RuntimeError: Cache not initialized and no config provided
FAILED tests/unit/test_cache_comprehensive.py::TestCacheErrorHandling::test_batch_operations_with_partial_failures - AssertionError: assert 'ETH/USDT' in {'ADA/USDT': None, 'BTC/USDT': {'price': 50000}}
FAILED tests/unit/test_cache_eviction.py::TestCacheEviction::test_evict_expired_entries - AssertionError: Expected 'delete' to have been called.
FAILED tests/unit/test_cache_eviction.py::TestCacheIntegration::test_memory_manager_integration - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
FAILED tests/unit/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_valid_config - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/unit/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_missing_sections - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/unit/test_core_security.py::TestDataProcessorSecurity::test_calculate_rsi_batch_data_validation - AssertionError: assert 'TEST' not in {'TEST':    open\n0   100\n1   101\n2   102}
FAILED tests/unit/test_core_security.py::TestMetricsEndpointSecurity::test_secure_defaults - assert False == True
FAILED tests/unit/test_coverage_enforcement.py::test_minimum_coverage_requirement - Failed: Coverage test timed out after 
10 minutes
FAILED tests/unit/test_coverage_enforcement.py::test_critical_modules_coverage - Failed: Critical module core.order_manager has insufficient test coverage
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_component_factory_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_cache_component_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_memory_manager_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_component_caching - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_configuration_override - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_async_component_operations - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_factory_global_instance - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestConfigurationIntegration::test_performance_tracker_config_integration 
- assert 1000.0 == 2000.0
FAILED tests/unit/test_dependency_injection.py::TestConfigurationPersistence::test_config_save_load - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_returns_200 - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_returns_json - AssertionError: assert 'status' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/health'}, 'm...
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_returns_metadata - KeyError: 'version'      
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_correlation_id_unique - KeyError: 'correlation_id'
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_timestamp_format - KeyError: 'timestamp'    
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_returns_200_when_healthy - assert 500 
in [200, 503]
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_returns_json_structure - AssertionError: assert 'ready' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'mes...
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_includes_all_check_components - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_correlation_id_unique - KeyError: 'correlation_id'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_timestamp_format - KeyError: 'timestamp'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_returns_503_when_bot_engine_unavailable - assert 500 == 503
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_check_details_structure - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_latency_measurement - KeyError: 'total_latency_ms'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[DATABASE_URL] - AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'me...
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[EXCHANGE_API_URL] - AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'me...
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[MESSAGE_QUEUE_URL] - AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 
'me...
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[REDIS_URL] - 
AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'me...  
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_bot_engine_check - KeyError: 'checks' 
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_exchange_check - KeyError: 'checks'   
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_cache_check - KeyError: 'checks'      
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_message_queue_check - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_database_check - KeyError: 'checks'   
FAILED tests/unit/test_endpoints.py::TestDashboardEndpoint::test_dashboard_endpoint_returns_200 - assert 500 == 200       
FAILED tests/unit/test_endpoints.py::TestDashboardEndpoint::test_dashboard_endpoint_returns_html - AssertionError: assert 
'text/html' in 'application/json'
FAILED tests/unit/test_endpoints.py::TestRateLimiting::test_rate_limit_headers_present - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestRateLimiting::test_dashboard_endpoint_not_rate_limited - assert 500 == 200       
FAILED tests/unit/test_endpoints.py::TestCORSSecurity::test_cors_allows_configured_origins - AssertionError: assert 'access-control-allow-origin' in Headers({'content-length': '48', 'content-type': 'application...
FAILED tests/unit/test_endpoints.py::TestCustomExceptionMiddleware::test_custom_exception_middleware_handles_exceptions - 
assert False
FAILED tests/unit/test_endpoints.py::TestTemplateRendering::test_dashboard_template_not_found - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestPrometheusMetrics::test_api_requests_counter_incremented - assert 429 == 200     
FAILED tests/unit/test_endpoints.py::TestRateLimitingEdgeCases::test_get_remote_address_exempt_function - AttributeError: 
'MockRequest' object has no attribute 'client'
FAILED tests/unit/test_endpoints.py::TestMiddlewareOrder::test_cors_middleware_configured - assert False
FAILED tests/unit/test_endpoints.py::TestMiddlewareOrder::test_rate_limit_middleware_configured - assert False
FAILED tests/unit/test_logging_and_resources.py::TestStructuredLogging::test_logger_initialization - AssertionError: assert <LogSensitivity.SECURE: 'secure'> == <LogSensitivity.INFO: 'info'>
FAILED tests/unit/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_context_manager - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have ...
FAILED tests/unit/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_initialization_failure - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have ...
FAILED tests/unit/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_close_error_handling - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have ...
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_context_manager_with_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have ... 
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_global_instance_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have ...      
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_endpoint_global_instance_cleanup - TypeError: object NoneType can't be used in 'await' expression
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cleanup_on_startup_failure - assert 
<Application 0x2891cfa03d0> is None
FAILED tests/unit/test_monitoring_observability.py::TestAlertingSystem::test_alert_deduplication - assert not True        
FAILED tests/unit/test_monitoring_observability.py::TestAlertingSystem::test_notification_delivery - AssertionError: Expected '_send_discord_notification' to have been called once. Called 0 times.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_paper_mode - assert None is not None        
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_backtest_mode - assert None is not None     
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_active - assert None is not None  
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_live_mode_with_retry - assert None is not None
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_trigger_counter - AssertionError: 
assert False
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_unknown_mode - AssertionError: Expected 'execute_paper_order' to be called once. Called 0 times.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_valid - ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_invalid_symbol_format - AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_negative_amount - AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_stop_without_loss - 
AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_invalid_signal_order_combo - AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_with_valid_payload - assert None is not NoneFAILED tests/unit/test_regression.py::TestRegression::test_network_error_retry_mechanism - AssertionError: Expected 'retry_async' to have been called once. Called 0 times.
FAILED tests/unit/test_regression.py::TestRegression::test_exchange_error_handling - AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
FAILED tests/unit/test_regression.py::TestRegression::test_timeout_error_recovery - AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
FAILED tests/unit/test_regression.py::TestRegression::test_memory_leak_prevention_in_long_running_sessions - assert False 
FAILED tests/unit/test_regression.py::TestRegression::test_invalid_trading_mode_fallback - assert None is not None        
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_api_key_sanitization - assert '***API_KEY_MASKED***' in 'Using API key: "sk-1234567890abcdef" for authentication'
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_financial_amount_sanitization - AssertionError: assert '***BALANCE_MASKED***' in '***12345.67_MASKED***, ***-987.65_MASKED***, ***11234.56_MASKED***'
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_audit_level_minimal_output - AssertionError: assert 'Security eve...y sk-123 used' == '[AUDIT] Secu... event logged'
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_complex_message_sanitization - assert '***API_KEY_MASKED***' in '\n        Processing trade for ***EMAIL_MASKED*** with API key "sk-1234567890abcdef...
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_structured_info_logging - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_structured_warning_logging - AssertionError: assert 'WARNING' in 'Alert triggered | alert_type=high_cpu | threshold=90.5\n'
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_structured_error_logging - AssertionError: assert 'ERROR' in 'Database connection failed | db_host=localhost | error_code=500\n'
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_debug_level_preserves_data - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_sensitivity_change - AssertionError: assert '***API_KEY_MASKED***' in 'Test message | api_key=sk-test123\n'
FAILED tests/unit/test_secure_logging.py::TestCoreLoggerIntegration::test_core_logger_sanitization - AssertionError: Expected 'info' to have been called once. Called 0 times.
FAILED tests/unit/test_secure_logging.py::TestLogSecurityVerification::test_no_api_keys_in_logs - AssertionError: assert 'sk-1234567890abcdef' not in 'Using API k...567890abcdef'
FAILED tests/unit/test_secure_logging.py::TestLogSecurityVerification::test_no_financial_amounts_in_logs - AssertionError: assert '12345.67' not in '***12345.67_MASKED***'
FAILED tests/unit/test_secure_logging.py::TestLogSecurityVerification::test_log_structure_preservation - AssertionError: assert '***AMOUNT_MASKED***' in 'Processing trade for ***EMAIL_MASKED*** with amount 12345.67 and API...
FAILED tests/unit/test_signal_router.py::TestJournalWriter::test_append_with_event_loop - AssertionError: Expected 'create_task' to have been called once. Called 0 times.
FAILED tests/unit/test_signal_router.py::TestJournalWriter::test_stop_method - AssertionError: Expected mock to have been 
awaited once. Awaited 0 times.
FAILED tests/utils/test_logger.py::TestJSONFormatter::test_format_record_with_exception - NameError: name 'sys' is not defined
FAILED tests/utils/test_logger.py::TestEnvironmentVariables::test_setup_logging_with_env_log_format_json - assert 2 == 1
FAILED tests/utils/test_logger.py::TestEnvironmentVariables::test_setup_logging_with_env_log_format_pretty - assert 2 == 1FAILED tests/utils/test_logger.py::TestEnvironmentVariables::test_setup_logging_with_env_log_format_color - assert 2 == 1 
FAILED tests/utils/test_logger.py::TestCorrelationIdSupport::test_generate_request_id_format - AssertionError: assert 20 == 21
FAILED tests/utils/test_logger.py::TestLoggerIntegration::test_structured_logging_integration - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
ERROR tests/acceptance/test_slo.py::TestSLOValidation::test_signal_latency_slo_median - TypeError: MetricsCollector.__init__() missing 1 required positional argument: 'config'
ERROR tests/acceptance/test_slo.py::TestSLOValidation::test_signal_latency_slo_p95 - TypeError: MetricsCollector.__init__() missing 1 required positional argument: 'config'
ERROR tests/acceptance/test_slo.py::TestSLOValidation::test_order_failure_rate_slo - TypeError: MetricsCollector.__init__() missing 1 required positional argument: 'config'
ERROR tests/data/test_data.py::test_historical_loader - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data.py::test_historical_data_validation - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data.py::test_data_resampling - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data.py::test_historical_data_pagination - data.historical_loader.ConfigurationError: data_dir contains invalid characters: .test_historical_data
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_get_cache_key - data.data_fetcher.PathTraversalError: 
Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_empty - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_and_load_from_cache - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_to_cache_empty_dataframe - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_expired - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_full_workflow_with_caching - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_multiple_symbols_workflow - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_realtime_data_workflow - data.data_fetcher.PathTraversalError: Cache directory path resolves outside allowed area: C:\Users\TU\AppData\Local\T...
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_save_to_cache_success - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_load_from_cache_success - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_load_from_cache_nonexistent - KeyError: 'name'        
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_load_from_cache_expired - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestCaching::test_cache_key_generation - KeyError: 'name'
ERROR tests/data/test_data_module_refactoring.py::TestVersioning::test_create_version_success - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/data/test_data_module_refactoring.py::TestVersioning::test_create_version_invalid_name - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/data/test_data_module_refactoring.py::TestVersioning::test_migrate_legacy_dataset - TypeError: expected str, bytes or os.PathLike object, not dict
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_full_data_loading_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\tm...
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_data_validation_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\tm...
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_cache_key_generation_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\tm...      
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_timeframe_utilities_workflow - data.historical_loader.ConfigurationError: Path separators not allowed in data_dir: C:\Users\TU\AppData\Local\Temp\tm...       
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_model_monitor_initialization - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_update_predictions - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_calculate_performance_metrics - _pickle.PicklingError: Can't 
pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_detect_drift_no_reference - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_detect_drift_with_reference - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_check_model_health - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_trigger_alert - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_save_monitoring_data - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_generate_report - _pickle.PicklingError: Can't pickle <class 
'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_start_stop_monitoring - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_monitoring_loop - _pickle.PicklingError: Can't pickle <class 
'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Magi...
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_get_secret_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_get_secret_not_found - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_store_secret_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_rotate_key - TypeError: VaultKeyManager.__init__() 
got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_health_check_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/test_integration.py::TestOptimizationIntegration::test_end_to_end_optimization_workflow - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_r...
ERROR tests/test_integration.py::TestOptimizationIntegration::test_component_interaction_data_to_backtest - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_r...
ERROR tests/test_integration.py::TestOptimizationIntegration::test_error_scenario_data_loading_failure - TypeError: Can't 
instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_r...

ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_single_return - ValueErroror: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_profit_factor_edge_cases - ValueError: : Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_total_return_percentage_safe_division - - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra in...      
ERROR tests/unit/test_cache_eviction.py::TestCacheIntegration::test_end_to_end_eviction_workflow
================== 427 failed, 2524 passed, 11 skipped, 251 warnings, 51 errors in 1392.04s (0:23:12) ===================        
(venv) PS C:\Users\TU\Desktop\new project\N1V1>
