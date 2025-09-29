(venv) C:\Users\TU\Desktop\new project\N1V1>pytest
========================================================================= test session starts =========================================================================
collected 2944 items

demo\test_anomaly_integration.py ..                                                                                                                              [  0%]
demo\test_simple_metrics.py ..                                                                                                                                   [  0%]
scripts\test_binary_labels.py .                                                                                                                                  [  0%]
test_logging_demo.py ...                                                                                                                                         [  0%]
tests\acceptance\test_docs.py .........                                                                                                                          [  0%]
tests\acceptance\test_ml_quality.py ......                                                                                                                       [  0%]
tests\acceptance\test_scalability.py ......                                                                                                                      [  0%]
tests\acceptance\test_slo.py .......                                                                                                                             [  1%]
tests\acceptance\test_stability.py ......                                                                                                                        [  1%]
tests\api\test_api_app.py ...................                                                                                                                    [  2%]
tests\api\test_endpoints.py .......................................................................................                                              [  5%]
tests\backtest\test_backtest_executor.py .....................                                                                                                   [  5%]
tests\backtest\test_backtester.py ............................................................                                                                   [  7%]
tests\backtest\test_regime_aware_backtester.py ...............                                                                                                   [  8%]
tests\core\test_alert_rules_manager.py .....................................                                                                                     [  9%]
tests\core\test_alerting.py ..................                                                                                                                   [ 10%]
tests\core\test_algorithmic_correctness.py ...............                                                                                                       [ 10%]
tests\core\test_binary_integration_enhanced.py ..............................                                                                                    [ 11%]
tests\core\test_binary_model_metrics.py .............................................                                                                            [ 13%]
tests\core\test_bot_engine_comprehensive.py ................................................................................                                     [ 15%]
tests\core\test_cache_comprehensive.py ...................................................................                                                       [ 18%]
tests\core\test_circuit_breaker.py .........................                                                                                                     [ 19%]
tests\core\test_config_manager.py ......................................                                                                                         [ 20%]
tests\core\test_dashboard_manager.py ..................................                                                                                          [ 21%]
tests\core\test_dependency_injection.py ...................                                                                                                      [ 22%]
tests\core\test_diagnostics.py ...........................                                                                                                       [ 23%]
tests\core\test_ensemble_manager.py .......................................                                                                                      [ 24%]
tests\core\test_event_driven_architecture.py ..................................                                                                                  [ 25%]
tests\core\test_execution.py ..........................................                                                                                          [ 26%]
tests\core\test_journal_writer.py .....                                                                                                                          [ 27%]
tests\core\test_live_executor.py .................                                                                                                               [ 27%]
tests\core\test_logging_and_resources.py ................                                                                                                        [ 28%]
tests\core\test_monitoring_observability.py ...............................                                                                                      [ 29%]
tests\core\test_order_manager.py .........................................                                                                                       [ 30%]
tests\core\test_order_processor.py ................................                                                                                              [ 31%]
tests\core\test_paper_executor.py ...................................                                                                                            [ 32%]
tests\core\test_performance_optimization.py ........................                                                                                             [ 33%]
tests\core\test_regression.py ..............                                                                                                                     [ 34%]
tests\core\test_reliability_manager.py ..........................                                                                                                [ 35%]
tests\core\test_safe_mode.py .                                                                                                                                   [ 35%]
tests\core\test_self_healing_engine.py ..........................................................                                                                [ 37%]
tests\core\test_signal_router.py .........................                                                                                                       [ 38%]
tests\core\test_signal_router_facade.py ....F...                                                                                                                 [ 38%]
tests\core\test_task_manager.py .........                                                                                                                        [ 38%]
tests\core\test_timeframe_manager.py .........................                                                                                                   [ 39%]
tests\core\test_trading_signal_amount.py ......                                                                                                                  [ 39%]
tests\data\test_data.py ............                                                                                                                             [ 40%]
tests\data\test_data_fetcher.py ..............................................                                                                                   [ 41%]
tests\data\test_data_fixes.py ....s.sssss.........................ss........                                                                                     [ 43%]
tests\data\test_data_module_refactoring.py ............................                                                                                          [ 44%]
tests\data\test_data_security.py .........................................                                                                                       [ 45%]
tests\data\test_historical_loader.py ...................................                                                                                         [ 46%]
tests\execution\test_smart_layer.py ......................                                                                                                       [ 47%]
tests\execution\test_validator.py ................................                                                                                               [ 48%]
tests\integration\test_cross_feature_integration.py ......F.........                                                                                             [ 49%]
tests\integration\test_distributed_system.py .................                                                                                                   [ 49%]
tests\integration\test_ml_serving_integration.py ..........                                                                                                      [ 50%]
tests\integration\test_order_flow_integration.py .........                                                                                                       [ 50%]
tests\knowledge_base\test_knowledge_base.py .....................................................................                                                [ 52%]
tests\ml\test_features.py ................F...                                                                                                                   [ 53%]
tests\ml\test_indicators.py .......................                                                                                                              [ 54%]
tests\ml\test_ml.py ..................................................s                                                                                          [ 55%]
tests\ml\test_ml_artifact_model_card.py .                                                                                                                        [ 55%]
tests\ml\test_ml_filter.py .......ssss....................                                                                                                       [ 56%]
tests\ml\test_ml_signal_router.py ...                                                                                                                            [ 57%]
tests\ml\test_model_loader.py .....................                                                                                                              [ 57%]
tests\ml\test_model_monitor.py .................                                                                                                                 [ 58%]
tests\ml\test_predictive_models.py .....................                                                                                                         [ 59%]
.........                                                                                                              [ 61%]
tests\ml\test_trainer.py .......................................                                                                                                 [ 62%]
tests\notifier\test_discord_integration.py ....................                                                                                                  [ 63%]
tests\notifier\test_discord_notifier.py .....................................................                                                                    [ 65%]
tests\optimization\test_asset_selector.py .......................................                                                                                [ 66%]
tests\optimization\test_async_optimizer.py ..................s..........................................                                                         [ 68%]
tests\optimization\test_cross_asset_validation.py ..............................                                                                                 [ 69%]
tests\optimization\test_optimization.py sssss.......................................................................                                             [ 72%]
tests\optimization\test_walk_forward.py ........................                                                                                                 [ 73%]
tests\portfolio\test_allocation_engine.py ................                                                                                                       [ 73%]
tests\portfolio\test_portfolio.py ................                                                                                                               [ 74%]
tests\portfolio\test_strategy_ensemble.py .....................                                                                                                  [ 74%]
tests\risk\test_adaptive_policy.py ..........................                                                                                                    [ 75%]
tests\risk\test_adaptive_risk.py .......................                                                                                                         [ 76%]
tests\risk\test_anomaly_detection.py .........................                                                                                                   [ 77%]
tests\risk\test_anomaly_detector.py .................................                                                                                            [ 78%]
tests\risk\test_risk.py ...............................                                                                                                          [ 79%]
tests\risk\test_risk_manager_integration.py ............                                                                                                         [ 79%]
tests\security\test_core_security.py ..........                                                                                                                  [ 80%]
tests\security\test_key_management.py ..................                                                                                                         [ 80%]
tests\security\test_order_invariants.py ................................                                                                                         [ 82%]
tests\security\test_secret_manager.py .............................                                                                                              [ 83%]
tests\security\test_secure_logging.py ......................                                                                                                     [ 83%]
tests\strategies\test_market_regime.py ...................................                                                                                       [ 84%]
tests\strategies\test_regime_forecaster.py ...........................                                                                                           [ 85%]
tests\strategies\test_strategies.py .....................................s....                                                                                   [ 87%]
tests\strategies\test_strategy.py ...........                                                                                                                    [ 87%]
tests\strategies\test_strategy_generator.py ..........................                                                                                           [ 88%]
tests\strategies\test_strategy_integration.py .................                                                                                                  [ 89%]
tests\strategies\test_strategy_selector.py ..........F.................                                                                                          [ 90%]
tests\test_integration.py .............                                                                                                                          [ 90%]
tests\test_main.py ...............................                                                                                                               [ 91%]
tests\utils\test_adapter.py ......................                                                                                                               [ 92%]
tests\utils\test_circular_import_fix.py ...                                                                                                                      [ 92%]
tests\utils\test_config_loader.py ...............................................                                                                                [ 94%]
tests\utils\test_demo_time_utils.py ..................                                                                                                           [ 94%]
tests\utils\test_docstring_standardization.py ..................                                                                                                 [ 95%]
tests\utils\test_logger.py ...........................................................................                                                           [ 97%]
tests\utils\test_retry.py ..                                                                                                                                     [ 97%]
tests\utils\test_time.py ........................                                                                                                                [ 98%]
tests\utils\test_time_utils.py .......................................                                                                                           [100%]

============================================================================== FAILURES ===============================================================================
___________________________________________________________ TestSignalRouterFacade.test_facade_file_content ___________________________________________________________

self = <test_signal_router_facade.TestSignalRouterFacade object at 0x000002045E5FB130>

    def test_facade_file_content(self):
        """Test that the facade file has the expected content."""
        # Read the facade file directly
        with open("core/signal_router.py", "r") as f:
            content = f.read()

        # Check that it imports the expected classes
>       assert (
            "from .signal_router.router import SignalRouter, JournalWriter" in content
        )
E       assert 'from .signal_router.router import SignalRouter, JournalWriter' in '"""\ncore/signal_router.py\n\nFacade for the modular signal routing system.\n\nThis file provides backward compatibil...t JournalWriter, SignalRouter\n\n# Re-export for backward compatibility\n__all__ = ["SignalRouter", "JournalWriter"]\n'

tests\core\test_signal_router_facade.py:75: AssertionError
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-29 15:15:11 - slowapi - INFO - Storage has been reset and all limits cleared
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
INFO     slowapi:extension.py:360 Storage has been reset and all limits cleared
______________________________________ TestPerformanceCircuitBreakerIntegration.test_performance_metrics_circuit_breaker_impact _______________________________________

self = <test_cross_feature_integration.TestPerformanceCircuitBreakerIntegration object at 0x0000020469A58550>

    @pytest.mark.asyncio
    async def test_performance_metrics_circuit_breaker_impact(self):
        """Test that performance metrics capture circuit breaker impact."""
        # Setup monitoring
        monitor = RealTimePerformanceMonitor({})
        await monitor.start_monitoring()

        # Establish baseline performance
        baseline_metrics = []
        for i in range(10):
            with self.profiler.profile_function("baseline_operation"):
                time.sleep(0.001)
            baseline_metrics.append(self.profiler.metrics_history[-1].execution_time)

        baseline_avg = np.mean(baseline_metrics)

        # Trigger circuit breaker
        await self.cb.check_and_trigger({"equity": 8500})

        # Measure performance after circuit breaker
        post_cb_metrics = []
        for i in range(10):
            with self.profiler.profile_function("post_cb_operation"):
                time.sleep(0.001)
            post_cb_metrics.append(self.profiler.metrics_history[-1].execution_time)

        post_cb_avg = np.mean(post_cb_metrics)

        # Performance impact should be minimal (< 35% to account for monitoring overhead)
        impact = abs(post_cb_avg - baseline_avg) / baseline_avg
>       assert impact < 0.35, f"Circuit breaker caused {impact:.1%} performance impact"
E       AssertionError: Circuit breaker caused 39.1% performance impact
E       assert 0.3911274061536189 < 0.35

tests\integration\test_cross_feature_integration.py:315: AssertionError
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-29 15:15:39 - slowapi - INFO - Storage has been reset and all limits cleared
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
INFO     slowapi:extension.py:360 Storage has been reset and all limits cleared
__________________________________________________ TestFeatureValidation.test_validate_feature_importance_stability ___________________________________________________

self = <test_features.TestFeatureValidation object at 0x0000020469C8FF70>

    def test_validate_feature_importance_stability(self):
        """Test feature importance stability validation."""
        # This would require trained models, so we'll mock it
>       with patch("ml.features.train_test_split") as mock_split:

tests\ml\test_features.py:365:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <unittest.mock._patch object at 0x0000020473F49870>

    def get_original(self):
        target = self.getter()
        name = self.attribute

        original = DEFAULT
        local = False

        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True

        if name in _builtins and isinstance(target, ModuleType):
            self.create = True

        if not self.create and original is DEFAULT:
>           raise AttributeError(
                "%s does not have the attribute %r" % (target, name)
            )
E           AttributeError: <module 'ml.features' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\features.py'> does not have the attribute 'train_test_split'

..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: AttributeError
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-29 15:16:05 - slowapi - INFO - Storage has been reset and all limits cleared
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
INFO     slowapi:extension.py:360 Storage has been reset and all limits cleared
_________________________________________________________ TestRuleBasedSelector.test_select_strategy_sideways _________________________________________________________

args = (<test_strategy_selector.TestRuleBasedSelector object at 0x000002046A2F1930>,), keywargs = {}

    @wraps(func)
    def patched(*args, **keywargs):
>       with self.decoration_helper(patched,
                                    args,
                                    keywargs) as (newargs, newkeywargs):

..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <unittest.mock._patch object at 0x000002046A3778E0>

    def get_original(self):
        target = self.getter()
        name = self.attribute

        original = DEFAULT
        local = False

        try:
            original = target.__dict__[name]
        except (AttributeError, KeyError):
            original = getattr(target, name, DEFAULT)
        else:
            local = True

        if name in _builtins and isinstance(target, ModuleType):
            self.create = True

        if not self.create and original is DEFAULT:
>           raise AttributeError(
                "%s does not have the attribute %r" % (target, name)
            )
E           AttributeError: <module 'strategies.regime.strategy_selector' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\strategies\\regime\\strategy_selector.py'> does not have the attribute 'get_market_regime_detector'

..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: AttributeError
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
INFO     slowapi:extension.py:360 Storage has been reset and all limits cleared
========================================================================== warnings summary ===========================================================================
venv\lib\site-packages\starlette\formparsers.py:10
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\formparsers.py:10: PendingDeprecationWarning: Please use `import python_multipart` instead.
    import multipart

venv\lib\site-packages\pydantic\_internal\_fields.py:149
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pydantic\_internal\_fields.py:149: UserWarning: Field "model_version" has conflict with protected namespace "model_".

  You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================================================================= short test summary info =======================================================================
FAILED tests/core/test_signal_router_facade.py::TestSignalRouterFacade::test_facade_file_content - assert 'from .signal_router.router import SignalRouter, JournalWriter' in '"""\ncore/signal_router.py\n\nFacade for the modular signal routing system.\n\nThis file...
FAILED tests/integration/test_cross_feature_integration.py::TestPerformanceCircuitBreakerIntegration::test_performance_metrics_circuit_breaker_impact - AssertionError: Circuit breaker caused 39.1% performance impact
FAILED tests/ml/test_features.py::TestFeatureValidation::test_validate_feature_importance_stability - AttributeError: <module 'ml.features' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\features.py'> does not have the attribute 'train_test_split'
FAILED tests/strategies/test_strategy_selector.py::TestRuleBasedSelector::test_select_strategy_sideways - AttributeError: <module 'strategies.regime.strategy_selector' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\strategies\\regime\\strategy_selector.py'> does not ...
================================================= 4 failed, 2920 passed, 20 skipped, 2 warnings in 695.27s (0:11:35) ==================================================
sys:1: RuntimeWarning: coroutine '_cleanup_endpoint_on_exit' was never awaited
Object allocated at (most recent call last):
  File "<unknown>", lineno 0