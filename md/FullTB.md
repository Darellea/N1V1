FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_invalid_schema - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_invalid_symbol_format - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_negative_amount - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_stop_without_loss - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/core/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_invalid_signal_order_combo - Failed: DID NOT RAISE <class 'ValueError'>
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_with_validation_failure - AssertionError: assert 'filled' == 'validation_failed'
FAILED tests/data/test_data.py::test_data_validation - assert not True
FAILED tests/data/test_data.py::test_multiple_symbol_fetching - assert 0 == 2
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_initialize_success - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_success - assert 0 == 2
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_exchange_error - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_network_error - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_historical_data_with_caching - assert 0 == 1
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_realtime_data_tickers_only - AssertionError: assert 'BTC/USDT' in {}
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_get_realtime_data_with_orderbooks - AssertionError: assert 'BTC/USDT' in {}
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_fetch_ticker_success - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherAsyncMethods::test_fetch_orderbook_success - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherMultipleData::test_get_multiple_historical_data_success - AssertionError: assert 'BTC/USDT' in {}
FAILED tests/data/test_data_fetcher.py::TestDataFetcherMultipleData::test_get_multiple_historical_data_partial_failure - AssertionError: assert 'BTC/USDT' in {}
FAILED tests/data/test_data_fetcher.py::TestDataFetcherErrorScenarios::test_get_historical_data_unexpected_error - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherErrorScenarios::test_fetch_ticker_error - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherErrorScenarios::test_fetch_orderbook_error - core.api_protection.CircuitOpenError: Circuit is open for AsyncMock, aborting retry
FAILED tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_full_workflow_with_caching - assert 0 == 1
FAILED tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_multiple_symbols_workflow - assert 0 == 3
FAILED tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_realtime_data_workflow - assert 0 == 2
FAILED tests/data/test_memory_efficient_loading.py::TestMemoryEfficientLoading::test_large_dataset_loading_with_timeout - Failed: Large dataset loading timed out
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_with_validation - AssertionError: assert {'amount': 0.001, 'cost': 50.050000000000004, 'fee': {'cost': 0.050050000000000004, 'currenc...
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_error_handling_in_order_flow - AssertionError: assert {'amount': -1.0, 'cost': -50050.0, 'fee': {'cost': -50.05, 'currency': 'USDT'}, 'filled': -1...
FAILED tests/ml/test_indicators.py::TestOBV::test_calculate_obv_insufficient_data - assert False
FAILED tests/ml/test_realtime_model_monitoring.py::TestRealTimeModelMonitoring::test_realtime_drift_detection - assert False
FAILED tests/ml/test_realtime_model_monitoring.py::TestRealTimeModelMonitoring::test_streaming_metrics_accuracy - AssertionError: assert 'auc' in {}
FAILED tests/ml/test_realtime_model_monitoring.py::TestRealTimeModelMonitoring::test_alert_timing - assert 0 > 0
FAILED tests/ml/test_realtime_model_monitoring.py::TestRealTimeModelMonitoring::test_multiple_drift_algorithms - assert False
FAILED tests/ml/test_realtime_model_monitoring.py::TestRealTimeModelMonitoring::test_automated_retraining_trigger - assert False
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_python_random - assert [0.1096491303...50653676, ...] == [0.6394267984...74229113, ...]
FAILED tests/optimization/test_deterministic_optimization.py::TestDeterministicOptimization::test_optimization_timeout_safety - assert 78.52452087402344 < 30
36 failed, 3350 passed, 100 skipped, 1230 warnings in 1310.21s (0:21:50)