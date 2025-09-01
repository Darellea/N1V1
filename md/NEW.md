# Project Audit — TODO Checklist

## Warnings  
- [x] Fix async warning in tests/test_discord_notifier.py::TestDiscordNotifier::test_shutdown_with_bot (replace MagicMock with AsyncMock for `_bot_task` and ensure it is awaited).
- [ ] Fix RuntimeWarning in tests/test_order_manager.py::TestOrderManager::test_cancel_order_live_mode (properly await async mocks).  
- [ ] Fix RuntimeWarning in tests/test_task_manager.py::TestTaskManager::test_shutdown_prevents_new_tasks (await `asyncio.sleep` or patch with `AsyncMock`).  

## Skipped Tests  
- [ ] Investigate 16 skipped tests in tests/test_discord_integration.py due to missing environment variables.  
- [ ] Add configuration or environment setup to allow Discord integration tests to run OR mark them with `pytest.mark.integration` to exclude from default runs.  

## Coverage Gaps  
- [ ] Increase coverage for core/bot_engine.py (currently 31%). Add unit tests for main trading loop, shutdown, emergency shutdown, performance metrics update, and display update.  
- [ ] Add tests for core/signal_router.py (currently 31%). Mock strategies and verify proper signal routing and concurrency handling.  
- [ ] Add tests for core/execution/live_executor.py (currently 29%). Mock exchange interactions and ensure orders are submitted correctly.  
- [ ] Add tests for strategies/rsi_strategy.py and strategies/ema_cross_strategy.py (17% and 18%). Cover `generate_signals` logic and edge cases.  
- [ ] Add tests for utils/adapter.py (18%). Cover adapter conversion functions with representative sample inputs.  
- [ ] Add tests for ml/model_loader.py (13%). Ensure model loading, saving, and schema validation works with mock artifacts.  

## Refactoring / Reliability  
- [ ] Review and improve shutdown sequences in async components (bot_engine, task_manager, discord_bot) to ensure proper resource cleanup and awaited coroutines.  
- [ ] Add CLI tests for `--help` and `--status` flags if applicable to improve command line interface reliability.  

## Documentation  
- [ ] Document how to run tests with and without integration tests (e.g., using `pytest -m "not integration"`).  
- [ ] Update README.md with testing instructions, coverage targets, and environment variable requirements for integration tests.
