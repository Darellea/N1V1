======================================================================= short test summary info =======================================================================
FAILED tests/core/test_alert_rules_manager.py::TestAlertRule::test_alert_rule_evaluation_greater_equal - assert False
FAILED tests/core/test_alert_rules_manager.py::TestAlertRule::test_alert_rule_evaluation_less_equal - assert False
FAILED tests/core/test_alert_rules_manager.py::TestAlertRule::test_alert_rule_evaluation_with_deduplication - assert False
FAILED tests/core/test_alert_rules_manager.py::TestAlertRulesManager::test_evaluate_rules - assert 0 == 2
FAILED tests/core/test_alert_rules_manager.py::TestAlertRulesManager::test_evaluate_rules_with_deduplication - assert 0 == 1
FAILED tests/core/test_alert_rules_manager.py::TestAlertRuleEdgeCases::test_rule_with_float_values - assert False
FAILED tests/core/test_alert_rules_manager.py::TestAlertRulesManagerIntegration::test_multiple_rules_same_metric - assert 0 == 1
FAILED tests/core/test_alert_rules_manager.py::TestAlertRulesManagerIntegration::test_complex_metrics_evaluation - assert 0 == 1
FAILED tests/core/test_alert_rules_manager.py::TestAlertRulesManagerIntegration::test_deduplication_window_expiration - assert 0 == 1
FAILED tests/core/test_cache_comprehensive.py::TestCacheContextManager::test_context_manager_success - AssertionError: Expected 'close_cache' to have been called once. Called 0 times.
FAILED tests/core/test_config_manager.py::TestConfigManagerValidation::test_apply_safe_defaults_no_changes_needed - AssertionError: assert {'environment': {'mode': 'paper', 'debug': False, 'log_level': 'INFO'}, 'exchange': {'name': 'kucoin', 'sandbox': True, 'timeout': 30000, 'r...
FAILED tests/core/test_journal_writer.py::test_journal_writer_append_with_running_loop - AssertionError: assert False
FAILED tests/core/test_journal_writer.py::test_journal_writer_stop - AssertionError: assert False
FAILED tests/core/test_order_manager.py::TestOrderManager::test_execute_order_with_validation_failure - assert None is not None
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_buy_signal - AssertionError: assert Decimal('50000.00') < Decimal('0.1')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_sell_signal - AssertionError: assert Decimal('5400.00') < Decimal('0.1')
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_buy_order - TypeError: conversion from MagicMock to Decimal is not supported
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_with_dict_signal - ValueError: Signal price required for slippage calculation
FAILED tests/core/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_missing_price - TypeError: conversion from MagicMock to Decimal is not supported
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_complete_order_flow_paper_trading - assert None is not None
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_multiple_orders - assert False
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_order_flow_portfolio_mode - assert None is not None
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_balance_and_equity_calculation - assert None is not None
FAILED tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_concurrent_order_execution - assert 0 == 10
================================================= 36 failed, 2888 passed, 20 skipped, 2 warnings in 742.70s (0:12:22) =================================================