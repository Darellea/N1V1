=============================================== short test summary info ===============================================
FAILED tests/data/test_data.py::test_cache_operations - AssertionError: assert False
FAILED tests/data/test_data.py::test_historical_data_pagination - assert 2 == 4
FAILED tests/data/test_data_fetcher.py::TestDataFetcherInitialization::test_init_with_config - AssertionError: assert 'C:\\Users\\T...\\.test_cache' == '.test_cache'
FAILED tests/data/test_data_fetcher.py::TestDataFetcherInitialization::test_init_cache_directory_creation - data.data_fetcher.PathTraversalError: Invalid cache directory path
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_success - AssertionError: assert ['timestamp',...se', 'volume'] == ['open', 'hig...se', 'volume']
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_with_caching - data.data_fetcher.PathTraversalError: Invalid cache directory path
FAILED tests/data/test_data_fetcher.py::TestDataFetcherErrorScenarios::test_cache_operations_error_handling - data.data_fetcher.PathTraversalError: Invalid cache directory path
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_critical_cache_raises_exception_on_failure - AssertionError: CacheLoadError not raised
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_all_strategies_fail - AssertionError: <coroutine object DataFetcher._load_from_cache at 0x000002ABF3544D60> is not None
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_1_integer_ms - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_2_datetime_column - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timestamp_parsing_strategy_3_format_parsing - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataFetcherTimestampHandling::test_timezone_normalization - AttributeError: 'coroutine' object has no attribute 'empty'
FAILED tests/data/test_data_fixes.py::TestDataOptimizations::test_cache_save_optimization - data.data_fetcher.PathTraversalError: Invalid cache directory path
FAILED tests/data/test_data_module_refactoring.py::TestCaching::test_save_to_cache_success - AssertionError: assert False
FAILED tests/data/test_data_module_refactoring.py::TestCaching::test_load_from_cache_success - assert None is not None
FAILED tests/data/test_data_module_refactoring.py::TestVersioning::test_create_version_success - AssertionError: assert 'test_version_1_20250922_161051' is True
FAILED tests/data/test_data_module_refactoring.py::TestVersioning::test_migrate_legacy_dataset - AssertionError: assert 'migrated_version_20250922_161051' is True
FAILED tests/data/test_data_security.py::TestConfigurationValidation::test_validate_data_directory_absolute_paths - Failed: DID NOT RAISE <class 'data.historical_loader.ConfigurationError'>
FAILED tests/data/test_data_security.py::TestConfigurationValidation::test_setup_data_directory_with_malicious_config - Failed: DID NOT RAISE <class 'data.historical_loader.ConfigurationError'>
FAILED tests/data/test_data_security.py::TestConfigurationValidation::test_setup_data_directory_with_valid_config - AssertionError: assert False
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_execution_with_fallback - AssertionError: assert False
FAILED tests/execution/test_smart_layer.py::TestExecutionSmartLayer::test_adaptive_pricing_applied - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_small_limit_order_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_large_order_twap_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/execution/test_smart_layer.py::TestIntegrationScenarios::test_high_spread_dca_flow - AssertionError: assert <ExecutionStatus.FAILED: 'failed'> == <ExecutionStatus.COMPLETED: 'completed'>
FAILED tests/integration/test_ml_serving_integration.py::TestPerformanceBenchmarks::test_latency_under_load - AssertionError: 95th percentile latency too high: 231.97ms
FAILED tests/ml/test_ml_filter.py::TestMLFilter::test_save_load_model - assert False
FAILED tests/ml/test_ml_filter.py::TestFactoryFunctions::test_load_ml_filter - assert False
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_python_random - xgboost.core.XGBoostError: [16:11:58] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_numpy - xgboost.core.XGBoostError: [16:11:59] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_pandas - xgboost.core.XGBoostError: [16:11:59] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_lightgbm - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_xgboost - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_catboost - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_tensorflow - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_pytorch - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_packages - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_env_vars - AssertionError: assert '/usr/local/bin:/usr/bin' == '/usr/local/lib/python3.9'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_git_info - KeyError: 'commit_hash'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_git_failure - AssertionError: assert 'error' in {}
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_hardware - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_hardware_failure - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have th...
FAILED tests/ml/test_reproducibility.py::TestExperimentTrackerReproducibility::test_experiment_tracker_git_info_logging - KeyError: 'commit_hash'
FAILED tests/ml/test_reproducibility.py::TestReproducibilityValidation::test_deterministic_numpy_operations - xgboost.core.XGBoostError: [16:12:01] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityValidation::test_deterministic_pandas_operations - xgboost.core.XGBoostError: [16:12:01] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_full_reproducibility_workflow - xgboost.core.XGBoostError: [16:12:02] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_reproducibility_with_different_seeds - xgboost.core.XGBoostError: [16:12:03] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_reproducibility_with_same_seed - xgboost.core.XGBoostError: [16:12:03] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_serving.py::TestPredictionProcessing::test_process_single_prediction_success - TypeError: argument of type 'coroutine' is not iterable
FAILED tests/ml/test_serving.py::TestPredictionProcessing::test_process_single_prediction_model_error - Failed: DID NOT RAISE <class 'Exception'>
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_success - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_with_nan_values - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_with_zero_prices - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_outlier_removal - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestMainFunction::test_main_success - AssertionError: Expected 'exit' to not have been called. Called 1 times.
FAILED tests/ml/test_train.py::TestMainFunction::test_main_predictive_models_disabled - AssertionError: Expected 'exit' to not have been called. Called 1 times.
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_basic - xgboost.core.XGBoostError: [16:12:06] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_with_libs - xgboost.core.XGBoostError: [16:12:06] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_libs_unavailable - xgboost.core.XGBoostError: [16:12:06] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-075051481...
FAILED tests/ml/test_train.py::TestReproducibility::test_capture_environment_snapshot - AssertionError: assert 'x86_64' == 'Linux-5.4.0'
FAILED tests/ml/test_train.py::TestModelLoaderReproducibility::test_load_model_from_registry_success - ModuleNotFoundError: No module named 'mlflow'
FAILED tests/ml/test_train.py::TestModelLoaderReproducibility::test_load_model_from_registry_fallback - ModuleNotFoundError: No module named 'mlflow'
FAILED tests/ml/test_trainer.py::TestDataProcessing::test_load_data_success - NameError: name 'logger' is not defined
FAILED tests/ml/test_trainer.py::TestDataProcessing::test_load_data_file_not_found - NameError: name 'logger' is not defined
FAILED tests/optimization/test_asset_selector.py::TestAssetSelector::test_fetch_from_coingecko_async_success - AssertionError: assert None == {'ETH/USDT': 50000000000}
FAILED tests/optimization/test_cross_asset_validation.py::TestAssetSelector::test_apply_market_cap_weighting - assert 0.5 > 0.5
FAILED tests/optimization/test_optimization.py::TestWalkForwardOptimizer::test_optimization_with_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_optimization.py::TestGeneticOptimizer::test_population_initialization - AssertionError: assert 20 == 10
FAILED tests/optimization/test_optimization.py::TestGeneticOptimizer::test_population_initialization_no_bounds - assert 20 == 5
FAILED tests/optimization/test_optimization.py::TestCoreOptimizationAlgorithms::test_crossover_with_different_gene_counts - KeyError: 'param2'
FAILED tests/optimization/test_optimization.py::TestCoreOptimizationAlgorithms::test_parameter_validation_edge_cases - AssertionError: assert not True
FAILED tests/optimization/test_optimization.py::TestRLOptimizer::test_policy_save_load - assert 0 > 0
FAILED tests/optimization/test_optimization.py::TestOptimizerFactory::test_create_optimizer - assert 20 == 10
FAILED tests/optimization/test_optimization.py::TestCrossPairValidation::test_cross_pair_validation_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_walk_forward.py::TestWalkForwardOptimizer::test_optimize_no_windows - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_walk_forward.py::TestWalkForwardOptimizer::test_optimize_with_windows - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/optimization/test_walk_forward.py::TestIntegration::test_full_walk_forward_workflow - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/optimization/test_walk_forward.py::TestIntegration::test_error_handling_insufficient_data - NameError: name 'start_time' is not defined
FAILED tests/optimization/test_walk_forward.py::TestIntegration::test_error_handling_optimizer_failure - TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_normal - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required co...
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_high_volatility - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required co...
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_empty_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_none_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_market_monitor_error_handling - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required co...
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_calculation - assert False
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_fallback - AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x000002ABF6B3C...
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_position_sizing_method_selection - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_error_handling_in_calculations - TypeError: '>=' not supported between instances of 'coroutine' and 'int'
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskIntegration::test_complete_trade_workflow - assert False is True
FAILED tests/risk/test_anomaly_detector.py::TestPriceGapDetector::test_detect_normal_price_change - AssertionError: assert False is False
FAILED tests/risk/test_anomaly_detector.py::TestPriceGapDetector::test_detect_price_gap - AssertionError: assert True is True
FAILED tests/risk/test_anomaly_detector.py::TestPriceGapDetector::test_detect_small_gap - AssertionError: assert False is False
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
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_kill_switch_blocks_trading - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_defensive_mode_reduces_position_size - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_atr_position_sizing_with_multiplier - AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x000002ABF8272...
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
FAILED tests/strategies/test_strategy_generator.py::TestPerformance::test_generation_performance - assert None is not None
FAILED tests/strategies/test_strategy_integration.py::TestStrategyIntegrationWorkflow::test_strategy_error_handling_workflow - assert 0 >= 2
FAILED tests/strategies/test_strategy_integration.py::TestStrategyRiskManagerIntegration::test_risk_manager_position_sizing_integration - decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
FAILED tests/strategies/test_strategy_integration.py::TestStrategyDataFetcherIntegration::test_data_fetcher_error_recovery_integration - assert 1 == 2
FAILED tests/test_integration.py::TestOptimizationIntegration::test_output_validation_backtest_metrics - backtest.backtester.BacktestValidationError: Record 0: timestamp must be string
FAILED tests/test_integration.py::TestOptimizationIntegration::test_error_scenario_optimization_with_invalid_parameters - AttributeError: 'GeneticOptimizer' object has no attribute 'ParameterBounds'. Did you mean: 'parameter_bounds'?
FAILED tests/test_integration.py::TestOptimizationIntegration::test_large_dataset_handling - backtest.backtester.BacktestValidationError: Record 0: timestamp must be string
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
FAILED tests/unit/test_binary_integration.py::TestErrorRecovery::test_strategy_selection_error_recovery - AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == 'UNKNOWN'
FAILED tests/unit/test_binary_integration.py::TestErrorRecovery::test_complete_pipeline_error_recovery - AssertionError: assert 'Integration error' in 'Binary integration disabled'
FAILED tests/unit/test_binary_integration.py::TestPerformanceAndScalability::test_concurrent_market_data_processing - AssertionError: assert False == True
FAILED tests/unit/test_binary_integration.py::TestLoggingAndMonitoring::test_binary_prediction_logging - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationFeatureExtraction::test_calculate_macd_edge_cases - assert 0.02243589743588359 == 0.0
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationGlobalFunctions::test_get_binary_integration_with_config - assert False == True
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationMetricsIntegration::test_metrics_recording_in_prediction - AssertionError: Expected 'record_prediction' to have been called once. Called 0 times.
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_predict_binary_model_with_model_exception - assert -1.0 == 0.0
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationErrorRecovery::test_select_strategy_with_regime_detector_failure - AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == <MarketRegime.UNKNOWN: 'unknown'>
FAILED tests/unit/test_binary_integration_enhanced.py::TestBinaryModelIntegrationLogging::test_binary_prediction_logging_integration - AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
FAILED tests/unit/test_binary_model_metrics.py::TestMetricsCollection::test_collect_binary_model_metrics_with_exception - AssertionError: assert False
FAILED tests/unit/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_trade_frequency_drift - AssertionError: assert 'trade frequency changed' in 'Trade frequency changed by 100.0%'
FAILED tests/unit/test_binary_model_metrics.py::TestAlertGeneration::test_check_for_alerts_accuracy_drop - assert 0 == 1
FAILED tests/unit/test_binary_model_metrics.py::TestPerformanceReporting::test_get_performance_report - KeyError: 'was_correct'
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_collect_metrics_with_binary_integration_import_error - AttributeError: <module 'core.binary_model_metrics' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\binary_m...
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_record_prediction_with_invalid_data - TypeError: '>=' not supported between instances of 'str' and 'float'
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_record_decision_outcome_with_invalid_data - TypeError: unsupported operand type(s) for +=: 'float' and 'str'
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_calculate_metrics_with_corrupted_data - TypeError: '>' not supported between instances of 'str' and 'datetime.datetime'
FAILED tests/unit/test_binary_model_metrics.py::TestDataIntegrity::test_prediction_history_data_integrity - TypeError: BinaryModelMetricsCollector.record_prediction() got an unexpected keyword argument 'decision'
FAILED tests/unit/test_bot_engine.py::TestBotEngine::test_trading_cycle - AssertionError: Expected 'get_historical_data' to have been called once. Called 0 times.
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_minimal_config - AssertionError: assert [] == ['BTC/USDT']
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_portfolio_mode - AssertionError: assert set() == {'BTC/USDT', 'ETH/USDT'}
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_invalid_balance - assert 10000.0 == 1000.0
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineInitialization::test_initialization_with_empty_markets - AssertionError: assert [] == ['BTC/USDT']
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_calculate_max_drawdown - assert 0.061224489795918366 == 0.08
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_success - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEnginePerformanceTracking::test_record_trade_equity_missing_equity - AttributeError: 'NoneType' object has no attribute 'get_equity'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineStateManagement::test_run_main_loop_normal_operation - AssertionError: Expected '_update_display' to have been called once. Called 0 times.
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_process_binary_integration_enabled - assert 0 == 1
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineIntegration::test_execute_integrated_decisions - AssertionError: Expected 'execute_order' to have been called once. Called 0 times.
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_trading_cycle_with_data_fetch_error - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_signal_generation_with_strategy_error - Exception: Strategy failed
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_risk_evaluation_with_component_error - AttributeError: 'NoneType' object has no attribute 'evaluate_signal'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_order_execution_with_component_error - AttributeError: 'NoneType' object has no attribute 'execute_order'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineErrorHandling::test_state_update_with_component_errors - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_empty_market_data - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_trading_cycle_with_none_market_data - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_bot_engine_comprehensive.py::TestBotEngineEdgeCases::test_concurrent_trading_cycles - AttributeError: 'NoneType' object has no attribute 'get_balance'
FAILED tests/unit/test_cache_comprehensive.py::TestCacheBatchOperations::test_set_multiple_ohlcv_success - assert False == True
FAILED tests/unit/test_cache_comprehensive.py::TestCacheInvalidation::test_invalidate_symbol_data_specific_types - assert 3 == 1
FAILED tests/unit/test_cache_comprehensive.py::TestCacheContextManager::test_context_manager_success - RuntimeError: Cache not initialized and no config provided
FAILED tests/unit/test_cache_comprehensive.py::TestCacheErrorHandling::test_batch_operations_with_partial_failures - AssertionError: assert 'ETH/USDT' in {'ADA/USDT': None, 'BTC/USDT': {'price': 50000}}
FAILED tests/unit/test_cache_eviction.py::TestCacheEviction::test_evict_expired_entries - AssertionError: Expected 'delete' to have been called.
FAILED tests/unit/test_cache_eviction.py::TestCacheIntegration::test_memory_manager_integration - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
FAILED tests/unit/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_valid_config - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/unit/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_missing_sections - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/unit/test_core_security.py::TestDataProcessorSecurity::test_calculate_rsi_batch_data_validation - AssertionError: assert 'TEST' not in {'TEST':    open\n0   100\n1   101\n2   102}
FAILED tests/unit/test_core_security.py::TestMetricsEndpointSecurity::test_secure_defaults - assert False == True
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_get_cache_key - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_empty - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_and_load_from_cache - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_to_cache_empty_dataframe - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_expired - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_full_workflow_with_caching - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_multiple_symbols_workflow - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_realtime_data_workflow - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_model_monitor_initialization - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_update_predictions - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_calculate_performance_metrics - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_detect_drift_no_reference - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_detect_drift_with_reference - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_check_model_health - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_trigger_alert - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_save_monitoring_data - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_generate_report - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_start_stop_monitoring - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_monitoring_loop - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.Ma...
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_get_secret_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_get_secret_not_found - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_store_secret_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_rotate_key - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_health_check_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/test_integration.py::TestOptimizationIntegration::test_end_to_end_optimization_workflow - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get...
ERROR tests/test_integration.py::TestOptimizationIntegration::test_component_interaction_data_to_backtest - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get...
ERROR tests/test_integration.py::TestOptimizationIntegration::test_error_scenario_data_loading_failure - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_constant_returns - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_single_return - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_profit_factor_edge_cases - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_total_return_percentage_safe_division - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra ...
ERROR tests/unit/test_cache_eviction.py::TestCacheIntegration::test_end_to_end_eviction_workflow
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! KeyboardInterrupt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py:1116: KeyboardInterrupt
(to show a full traceback on KeyboardInterrupt use --full-trace)
================== 199 failed, 1818 passed, 12 skipped, 251 warnings, 32 errors in 803.68s (0:13:23) ==================