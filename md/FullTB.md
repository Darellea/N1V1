======================================================================= short test summary info =======================================================================
FAILED tests/core/test_dependency_injection.py::TestDependencyInjection::test_memory_manager_creation - NameError: name 'os' is not defined
FAILED tests/core/test_dependency_injection.py::TestConfigurationIntegration::test_memory_manager_config_integration - NameError: name 'os' is not defined
FAILED tests/core/test_dependency_injection.py::TestComponentIsolation::test_factory_creates_correct_types - NameError: name 'os' is not defined
FAILED tests/integration/test_distributed_system.py::TestDistributedTaskManager::test_task_enqueue_dequeue - AttributeError: 'async_generator' object has no attribute 'enqueue_signal_task'
FAILED tests/integration/test_distributed_system.py::TestDistributedTaskManager::test_worker_processing - AttributeError: 'async_generator' object has no attribute 'register_task_handler'
FAILED tests/integration/test_distributed_system.py::TestDistributedTaskManager::test_task_failure_retry - AttributeError: 'async_generator' object has no attribute 'register_task_handler'
FAILED tests/integration/test_distributed_system.py::TestDistributedTaskManager::test_queue_status_monitoring - AttributeError: 'async_generator' object has no attribute 'enqueue_signal_task'
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_signal_scheduling - AttributeError: 'async_generator' object has no attribute 'schedule_signal_processing'
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_backtest_scheduling - AttributeError: 'async_generator' object has no attribute 'schedule_backtest'
FAILED tests/integration/test_distributed_system.py::TestDistributedScheduler::test_optimization_scheduling - AttributeError: 'async_generator' object has no attribute 'schedule_optimization'
FAILED tests/integration/test_distributed_system.py::TestDistributedExecutor::test_task_handler_registration - AttributeError: 'async_generator' object has no attribute 'task_manager'
FAILED tests/integration/test_distributed_system.py::TestDistributedExecutor::test_signal_task_processing - AttributeError: 'async_generator' object has no attribute 'initialize'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_basic_notification - AttributeError: 'async_generator' object has no attribute 'send_notification'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_with_embed - AttributeError: 'async_generator' object has no attribute 'send_notification'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_trade_alert - AttributeError: 'async_generator' object has no attribute 'send_trade_alert'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_signal_alert - AttributeError: 'async_generator' object has no attribute 'send_signal_alert'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_error_alert - AttributeError: 'async_generator' object has no attribute 'send_error_alert'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_performance_report - AttributeError: 'async_generator' object has no attribute 'send_performance_report'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_rate_limit_handling - AttributeError: 'async_generator' object has no attribute 'send_notification'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_large_payload - AttributeError: 'async_generator' object has no attribute 'send_notification'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_special_characters - AttributeError: 'async_generator' object has no attribute 'send_notification'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_empty_content - AttributeError: 'async_generator' object has no attribute 'send_notification'
FAILED tests/notifier/test_discord_integration.py::TestDiscordIntegration::test_webhook_integration_order_failure_alert - AttributeError: 'async_generator' object has no attribute 'send_order_failure_alert'
=============================================== 23 failed, 3010 passed, 98 skipped, 1087 warnings in 707.53s (0:11:47) ================================================