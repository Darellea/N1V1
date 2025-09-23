(venv) C:\Users\TU\Desktop\new project\N1V1>pytest --tb=short
================================================= test session starts =================================================
platform win32 -- Python 3.10.11, pytest-8.0.0, pluggy-1.6.0
rootdir: C:\Users\TU\Desktop\new project\N1V1
configfile: pytest.ini
plugins: anyio-3.7.1, hypothesis-6.92.6, asyncio-0.23.5, cov-4.1.0, mock-3.12.0, xdist-3.5.0
asyncio: mode=auto
collected 3016 items

demo\test_anomaly_integration.py ..                                                                              [  0%]
demo\test_simple_metrics.py ..                                                                                   [  0%]
scripts\test_binary_labels.py .                                                                                  [  0%]
test_logging_demo.py ...                                                                                         [  0%]
tests\acceptance\test_docs.py .........                                                                          [  0%]
tests\acceptance\test_ml_quality.py ......                                                                       [  0%]
tests\acceptance\test_scalability.py ......                                                                      [  0%]
tests\acceptance\test_slo.py .......                                                                             [  1%]
tests\acceptance\test_stability.py ......                                                                                                                        [  1%]
tests\backtest\test_backtest_executor.py .....................                                                                                                   [  2%]
tests\backtest\test_backtester.py ............................................................                                                                   [  4%]
tests\backtest\test_regime_aware_backtester.py ...............                                                                                                   [  4%]
tests\data\test_data.py ...F......F.                                                                                                                             [  4%]
tests\data\test_data_fetcher.py F..F......F.........EEEEE.F....F...FFF.....EEE                                                                                   [  6%]
tests\data\test_data_fixes.py ....F.FFFFF..........F......................F.                                                                                     [  8%]
tests\data\test_data_module_refactoring.py .......FFFF.......F.F.F.....                                                                                          [  8%]
tests\data\test_data_security.py .F.FFFF..........F.FF.................FFF                                                                                       [ 10%]
tests\data\test_historical_loader.py ..F..FF..........F..........FFFEEEE                                                                                         [ 11%]
tests\execution\test_smart_layer.py ......................                                                                                                       [ 12%]
tests\execution\test_validator.py ................................                                                                                               [ 13%]
tests\integration\test_distributed_system.py .................                                                                                                   [ 13%]
tests\integration\test_ml_serving_integration.py ..........                                                                                                      [ 14%]
tests\integration\test_order_flow_integration.py .........                                                                                                       [ 14%]
tests\knowledge_base\test_knowledge_base.py .................................................................... [ 16%]
.                                                                                                                                                                [ 16%]
tests\ml\test_features.py ....................                                                                                                                   [ 17%]
tests\ml\test_indicators.py .......................                                                                                                              [ 18%]
tests\ml\test_ml.py ..................................................s                                                                                          [ 19%]
tests\ml\test_ml_artifact_model_card.py .                                                                                                                        [ 19%]
tests\ml\test_ml_filter.py .......ssss.........F....F.....                                                                                                       [ 20%]
tests\ml\test_ml_signal_router.py ...                                                                                                                            [ 21%]
tests\ml\test_model_loader.py .....................                                                                                                              [ 21%]
tests\ml\test_model_monitor.py EEEEEEEEEEE......                                                                                                                 [ 22%]
tests\ml\test_predictive_models.py .....................                                                                                                         [ 22%]
tests\ml\test_reproducibility.py Windows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 39 in test_set_deterministic_seeds_python_random
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FWindows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 54 in test_set_deterministic_seeds_numpy
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FWindows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 69 in test_set_deterministic_seeds_pandas
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FFFFFF.FFFFFF...F.Windows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 390 in test_deterministic_numpy_operations
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FWindows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 405 in test_deterministic_pandas_operations
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
F..Windows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 477 in test_full_reproducibility_workflow
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FWindows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 518 in test_reproducibility_with_different_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FWindows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 531 in test_reproducibility_with_same_seed
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
F                                                                                                     [ 23%]
tests\ml\test_serving.py ...........FF....                                                                                                                       [ 24%]
tests\ml\test_train.py ......F.FFF..Windows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 898 in initialize_experiment_tracking
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 962 in main
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_train.py", line 298 in test_main_success
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py", line 1379 in patched
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FF....Windows fatal exception: access violation

Thread 0x00004308 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000038dc (most recent call first):
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 1176 in create_module
  File "<frozen importlib._bootstrap>", line 571 in module_from_spec
  File "<frozen importlib._bootstrap>", line 674 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\python\pywrap_tensorflow.py", line 73 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap>", line 1078 in _handle_fromlist
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tensorflow\__init__.py", line 40 in <module>
  File "<frozen importlib._bootstrap>", line 241 in _call_with_frames_removed
  File "<frozen importlib._bootstrap_external>", line 883 in exec_module
  File "<frozen importlib._bootstrap>", line 688 in _load_unlocked
  File "<frozen importlib._bootstrap>", line 1006 in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 1027 in _find_and_load
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 113 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_train.py", line 477 in test_set_deterministic_seeds_basic
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py", line 1379 in patched
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 193 in pytest_pyfunc_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py", line 1836 in runtest
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 173 in pytest_runtest_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 266 in <lambda>
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 345 in from_call
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 265 in call_runtest_hook
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 226 in call_and_report
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 133 in runtestprotocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\runner.py", line 114 in pytest_runtest_protocol
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 351 in pytest_runtestloop
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 326 in _main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 272 in wrap_session
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\main.py", line 319 in pytest_cmdline_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_callers.py", line 121 in _multicall
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_manager.py", line 120 in _hookexec
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pluggy\_hooks.py", line 512 in __call__
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 174 in main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\config\__init__.py", line 197 in console_main
  File "C:\Users\TU\Desktop\new project\N1V1\venv\Scripts\pytest.exe\__main__.py", line 7 in <module>
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86 in _run_code
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196 in _run_module_as_main
FFFF..F.F                                                                                                              [ 25%]
tests\ml\test_trainer.py ..............FF.......................                                                                                                 [ 26%]
tests\notifier\test_discord_integration.py ....................                                                                                                  [ 27%]
tests\notifier\test_discord_notifier.py .....................................................                                                                    [ 29%]
tests\optimization\test_asset_selector.py .....................F.................                                                                                [ 30%]
tests\optimization\test_cross_asset_validation.py ............F.................                                                                                 [ 31%]
tests\optimization\test_optimization.py sssss...F...FF........................F.......F........FF............... [ 33%]
F...                                                                                                                                                             [ 33%]
tests\optimization\test_walk_forward.py .........FF..........FFF                                                                                                 [ 34%]
tests\portfolio\test_allocation_engine.py ................                                                                                                       [ 35%]
tests\portfolio\test_portfolio.py ................                                                                                                               [ 35%]
tests\portfolio\test_strategy_ensemble.py .....................                                                                                                  [ 36%]
tests\risk\test_adaptive_policy.py .FF..................FF..F                                                                                                    [ 37%]
tests\risk\test_adaptive_risk.py FF................FF.F.                                                                                                         [ 38%]
tests\risk\test_anomaly_detector.py .........FFF.F....FFFFF.....F..FF                                                                                            [ 39%]
tests\risk\test_risk.py ...........F.F....F..F..F...FF.                                                                                                          [ 40%]
tests\risk\test_risk_manager_integration.py FFFFFFFFFF..                                                                                                         [ 40%]
tests\security\test_key_management.py EEEEEFF.....F.....                                                                                                         [ 41%]
tests\security\test_order_invariants.py ................................                                                                                         [ 42%]
tests\security\test_secret_manager.py ......F..F................FFF                                                                                              [ 43%]
tests\strategies\test_market_regime.py ...................................                                                                                       [ 44%]
tests\strategies\test_regime_forecaster.py ...........................                                                                                           [ 45%]
tests\strategies\test_strategies.py .................FF..................s...F                                                                                   [ 46%]
tests\strategies\test_strategy.py ...........                                                                                                                    [ 47%]
tests\strategies\test_strategy_generator.py ...........F........F.....                                                                                           [ 47%]
tests\strategies\test_strategy_integration.py ..F..F...F.......                                                                                                  [ 48%]
tests\strategies\test_strategy_selector.py ............................                                                                                          [ 49%]
tests\test_integration.py EE..FE.F.....F                                                                                                                         [ 49%]
tests\test_main.py ...............................                                                                                                               [ 50%]
tests\unit\test_alert_rules_manager.py .....................................                                                                                     [ 52%]
tests\unit\test_alerting.py ..................                                                                                                                   [ 52%]
tests\unit\test_algorithmic_correctness.py EEEEFFF..FFF..F                                                                                                       [ 53%]
tests\unit\test_anomaly_detection.py .........................                                                                                                   [ 54%]
tests\unit\test_api_app.py .F...........F.....                                                                                                                   [ 54%]
tests\unit\test_async_optimizer.py ..................s..........................................                                                                 [ 56%]
tests\unit\test_binary_integration.py ............FF.................F...FFFF.F....                                                                              [ 58%]
tests\unit\test_binary_integration_enhanced.py ........F....F.F.FF......F....                                                                                    [ 59%]
tests\unit\test_binary_model_metrics.py .........................FFF...F....FFFF.F...                                                                            [ 60%]
tests\unit\test_bot_engine.py ...F.........................                                                                                                      [ 61%]
tests\unit\test_bot_engine_comprehensive.py FFFF.......................F.FF...F......F.F.FFFF.F.FF..F......                                                      [ 63%]
tests\unit\test_cache_comprehensive.py .....................F...F.......F....F.......                                                                            [ 65%]
tests\unit\test_cache_eviction.py ..F................FE                                                                                                          [ 65%]
tests\unit\test_circuit_breaker.py .........................                                                                                                     [ 66%]
tests\unit\test_config_manager.py ................F..F..................                                                                                         [ 68%]
tests\unit\test_core_security.py .F..F.....                                                                                                                      [ 68%]
tests\unit\test_cross_feature_integration.py ...F............                                                                                                    [ 68%]
tests\unit\test_dashboard_manager.py ..................................                                                                                          [ 69%]
tests\unit\test_dependency_injection.py ..FFFFF.FF..F....F.                                                                                                      [ 70%]
tests\unit\test_diagnostics.py ...........................                                                                                                       [ 71%]
tests\unit\test_endpoints.py FFFFFFFFFFFFFFFFFFFFFF................FF........F...FF........F........FF.......F.. [ 74%]
.FF.                                                                                                                                                             [ 74%]
tests\unit\test_ensemble_manager.py .......................................                                                                                      [ 75%]
tests\unit\test_event_driven_architecture.py ..................................                                                                                  [ 76%]
tests\unit\test_execution.py ..........................................                                                                                          [ 78%]
tests\unit\test_journal_writer.py .....                                                                                                                          [ 78%]
tests\unit\test_live_executor.py .................                                                                                                               [ 78%]
tests\unit\test_logging_and_resources.py F...FFF....F.FFF                                                                                                        [ 79%]
tests\unit\test_monitoring_observability.py .............F.........FF......                                                                                      [ 80%]
tests\unit\test_order_manager.py F.FFFF............FF.............F.FFFF.F                                                                                       [ 81%]
tests\unit\test_order_processor.py ................................                                                                                              [ 82%]
tests\unit\test_paper_executor.py .........FF.....F......FFF.....F...                                                                                            [ 84%]
tests\unit\test_performance_optimization.py ........................                                                                                             [ 84%]
tests\unit\test_regression.py FFF.......F.F.                                                                                                                     [ 85%]
tests\unit\test_reliability_manager.py ..........................                                                                                                [ 86%]
tests\unit\test_safe_mode.py .                                                                                                                                   [ 86%]
tests\unit\test_secure_logging.py F.F..FFFFFFF.F..FF.F..                                                                                                         [ 86%]
tests\unit\test_self_healing_engine.py ..........................................................                                                                [ 88%]
tests\unit\test_signal_router.py ....FF...................                                                                                                       [ 89%]
tests\unit\test_signal_router_facade.py ........                                                                                                                 [ 89%]
tests\unit\test_task_manager.py .........                                                                                                                        [ 90%]
tests\unit\test_timeframe_manager.py .........................                                                                                                   [ 91%]
tests\unit\test_trading_signal_amount.py ......                                                                                                                  [ 91%]
tests\unit\test_types.py ..............                                                                                                                          [ 91%]
tests\utils\test_adapter.py ......................                                                                                                               [ 92%]
tests\utils\test_circular_import_fix.py ...                                                                                                                      [ 92%]
tests\utils\test_config_loader.py ...............................................                                                                                [ 94%]
tests\utils\test_demo_time_utils.py ..................                                                                                                           [ 94%]
tests\utils\test_docstring_standardization.py ..................                                                                                                 [ 95%]
tests\utils\test_logger.py ........................................................F.....FFFF........F                                                           [ 97%]
tests\utils\test_retry.py ..                                                                                                                                     [ 97%]
tests\utils\test_time.py ........................                                                                                                                [ 98%]
tests\utils\test_time_utils.py .......................................                                                                                           [100%]

=============================================================================== ERRORS ================================================================================
_____________________________________________________ ERROR at setup of TestDataFetcherCaching.test_get_cache_key _____________________________________________________
tests\data\test_data_fetcher.py:403: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
_________________________________________________ ERROR at setup of TestDataFetcherCaching.test_load_from_cache_empty _________________________________________________
tests\data\test_data_fetcher.py:403: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
_______________________________________________ ERROR at setup of TestDataFetcherCaching.test_save_and_load_from_cache ________________________________________________
tests\data\test_data_fetcher.py:403: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
_____________________________________________ ERROR at setup of TestDataFetcherCaching.test_save_to_cache_empty_dataframe _____________________________________________
tests\data\test_data_fetcher.py:403: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
________________________________________________ ERROR at setup of TestDataFetcherCaching.test_load_from_cache_expired ________________________________________________
tests\data\test_data_fetcher.py:403: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
____________________________________________ ERROR at setup of TestDataFetcherIntegration.test_full_workflow_with_caching _____________________________________________
tests\data\test_data_fetcher.py:791: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
_____________________________________________ ERROR at setup of TestDataFetcherIntegration.test_multiple_symbols_workflow _____________________________________________
tests\data\test_data_fetcher.py:791: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
______________________________________________ ERROR at setup of TestDataFetcherIntegration.test_realtime_data_workflow _______________________________________________
tests\data\test_data_fetcher.py:791: in setup_method
    self.fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
________________________________________ ERROR at setup of TestHistoricalDataLoaderIntegration.test_full_data_loading_workflow ________________________________________
tests\data\test_historical_loader.py:662: in setup_method
    self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)
data\historical_loader.py:63: in __init__
    self._setup_data_directory()
data\historical_loader.py:117: in _setup_data_directory
    self.data_dir = self._validate_data_directory(raw_data_dir)
data\historical_loader.py:86: in _validate_data_directory
    raise ConfigurationError("Absolute path detected")
E   data.historical_loader.ConfigurationError: Absolute path detected
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-23 10:03:21 - data.historical_loader - ERROR - Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmpmv65xgmg
2025-09-23 10:03:21 - data.historical_loader - ERROR - Data directory validation failed: Absolute path detected
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
ERROR    data.historical_loader:historical_loader.py:85 Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmpmv65xgmg
ERROR    data.historical_loader:historical_loader.py:119 Data directory validation failed: Absolute path detected
_________________________________________ ERROR at setup of TestHistoricalDataLoaderIntegration.test_data_validation_workflow _________________________________________
tests\data\test_historical_loader.py:662: in setup_method
    self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)
data\historical_loader.py:63: in __init__
    self._setup_data_directory()
data\historical_loader.py:117: in _setup_data_directory
    self.data_dir = self._validate_data_directory(raw_data_dir)
data\historical_loader.py:86: in _validate_data_directory
    raise ConfigurationError("Absolute path detected")
E   data.historical_loader.ConfigurationError: Absolute path detected
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-23 10:03:21 - data.historical_loader - ERROR - Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmptsf_185h
2025-09-23 10:03:21 - data.historical_loader - ERROR - Data directory validation failed: Absolute path detected
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
ERROR    data.historical_loader:historical_loader.py:85 Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmptsf_185h
ERROR    data.historical_loader:historical_loader.py:119 Data directory validation failed: Absolute path detected
______________________________________ ERROR at setup of TestHistoricalDataLoaderIntegration.test_cache_key_generation_workflow _______________________________________
tests\data\test_historical_loader.py:662: in setup_method
    self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)
data\historical_loader.py:63: in __init__
    self._setup_data_directory()
data\historical_loader.py:117: in _setup_data_directory
    self.data_dir = self._validate_data_directory(raw_data_dir)
data\historical_loader.py:86: in _validate_data_directory
    raise ConfigurationError("Absolute path detected")
E   data.historical_loader.ConfigurationError: Absolute path detected
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-23 10:03:22 - data.historical_loader - ERROR - Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmph_vod4f1
2025-09-23 10:03:22 - data.historical_loader - ERROR - Data directory validation failed: Absolute path detected
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
ERROR    data.historical_loader:historical_loader.py:85 Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmph_vod4f1
ERROR    data.historical_loader:historical_loader.py:119 Data directory validation failed: Absolute path detected
_______________________________________ ERROR at setup of TestHistoricalDataLoaderIntegration.test_timeframe_utilities_workflow _______________________________________
tests\data\test_historical_loader.py:662: in setup_method
    self.loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)
data\historical_loader.py:63: in __init__
    self._setup_data_directory()
data\historical_loader.py:117: in _setup_data_directory
    self.data_dir = self._validate_data_directory(raw_data_dir)
data\historical_loader.py:86: in _validate_data_directory
    raise ConfigurationError("Absolute path detected")
E   data.historical_loader.ConfigurationError: Absolute path detected
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-23 10:03:22 - data.historical_loader - ERROR - Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmpn45u15j0
2025-09-23 10:03:22 - data.historical_loader - ERROR - Data directory validation failed: Absolute path detected
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
ERROR    data.historical_loader:historical_loader.py:85 Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmpn45u15j0
ERROR    data.historical_loader:historical_loader.py:119 Data directory validation failed: Absolute path detected
________________________________________________ ERROR at setup of TestModelMonitor.test_model_monitor_initialization _________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
_____________________________________________________ ERROR at setup of TestModelMonitor.test_update_predictions ______________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
________________________________________________ ERROR at setup of TestModelMonitor.test_calculate_performance_metrics ________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
__________________________________________________ ERROR at setup of TestModelMonitor.test_detect_drift_no_reference __________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
_________________________________________________ ERROR at setup of TestModelMonitor.test_detect_drift_with_reference _________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
_____________________________________________________ ERROR at setup of TestModelMonitor.test_check_model_health ______________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
________________________________________________________ ERROR at setup of TestModelMonitor.test_trigger_alert ________________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
____________________________________________________ ERROR at setup of TestModelMonitor.test_save_monitoring_data _____________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
_______________________________________________________ ERROR at setup of TestModelMonitor.test_generate_report _______________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
____________________________________________________ ERROR at setup of TestModelMonitor.test_start_stop_monitoring ____________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
_______________________________________________________ ERROR at setup of TestModelMonitor.test_monitoring_loop _______________________________________________________
tests\ml\test_model_monitor.py:54: in setup_method
    joblib.dump(self.mock_model, self.model_path)
venv\lib\site-packages\joblib\numpy_pickle.py:553: in dump
    NumpyPickler(f, protocol=protocol).dump(value)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:487: in dump
    self.save(obj)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:603: in save
    self.save_reduce(obj=obj, *rv)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:687: in save_reduce
    save(cls)
venv\lib\site-packages\joblib\numpy_pickle.py:355: in save
    return Pickler.save(self, obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:560: in save
    f(self, obj)  # Call unbound method with explicit self
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1129: in save_type
    return self.save_global(obj)
..\..\..\AppData\Local\Programs\Python\Python310\lib\pickle.py:1076: in save_global
    raise PicklingError(
E   _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
____________________________________________________ ERROR at setup of TestVaultKeyManager.test_get_secret_success ____________________________________________________
tests\security\test_key_management.py:34: in vault_manager
    return VaultKeyManager(**vault_config)
E   TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
___________________________________________________ ERROR at setup of TestVaultKeyManager.test_get_secret_not_found ___________________________________________________
tests\security\test_key_management.py:34: in vault_manager
    return VaultKeyManager(**vault_config)
E   TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
___________________________________________________ ERROR at setup of TestVaultKeyManager.test_store_secret_success ___________________________________________________
tests\security\test_key_management.py:34: in vault_manager
    return VaultKeyManager(**vault_config)
E   TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
________________________________________________________ ERROR at setup of TestVaultKeyManager.test_rotate_key ________________________________________________________
tests\security\test_key_management.py:34: in vault_manager
    return VaultKeyManager(**vault_config)
E   TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
___________________________________________________ ERROR at setup of TestVaultKeyManager.test_health_check_success ___________________________________________________
tests\security\test_key_management.py:34: in vault_manager
    return VaultKeyManager(**vault_config)
E   TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
_________________________________________ ERROR at setup of TestOptimizationIntegration.test_end_to_end_optimization_workflow _________________________________________
tests\test_integration.py:136: in mock_data_fetcher
    return MockDataFetcher()
E   TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_realtime_data, shutdown
______________________________________ ERROR at setup of TestOptimizationIntegration.test_component_interaction_data_to_backtest ______________________________________
tests\test_integration.py:136: in mock_data_fetcher
    return MockDataFetcher()
E   TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_realtime_data, shutdown
_______________________________________ ERROR at setup of TestOptimizationIntegration.test_error_scenario_data_loading_failure ________________________________________
tests\test_integration.py:136: in mock_data_fetcher
    return MockDataFetcher()
E   TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_realtime_data, shutdown
_______________________________________ ERROR at setup of TestPerformanceTrackerCorrectness.test_sharpe_ratio_constant_returns ________________________________________
tests\unit\test_algorithmic_correctness.py:29: in setup_method
    self.tracker = PerformanceTracker(self.config)
core\performance_tracker.py:42: in __init__
    config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr setup ------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
_________________________________________ ERROR at setup of TestPerformanceTrackerCorrectness.test_sharpe_ratio_single_return _________________________________________
tests\unit\test_algorithmic_correctness.py:29: in setup_method
    self.tracker = PerformanceTracker(self.config)
core\performance_tracker.py:42: in __init__
    config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr setup ------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
__________________________________________ ERROR at setup of TestPerformanceTrackerCorrectness.test_profit_factor_edge_cases __________________________________________
tests\unit\test_algorithmic_correctness.py:29: in setup_method
    self.tracker = PerformanceTracker(self.config)
core\performance_tracker.py:42: in __init__
    config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr setup ------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
___________________________________ ERROR at setup of TestPerformanceTrackerCorrectness.test_total_return_percentage_safe_division ____________________________________
tests\unit\test_algorithmic_correctness.py:29: in setup_method
    self.tracker = PerformanceTracker(self.config)
core\performance_tracker.py:42: in __init__
    config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr setup ------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
______________________________________________ ERROR at setup of TestCacheIntegration.test_end_to_end_eviction_workflow _______________________________________________
file C:\Users\TU\Desktop\new project\N1V1\tests\unit\test_cache_eviction.py, line 378
      @pytest.mark.asyncio
      async def test_end_to_end_eviction_workflow(self, cache_instance, mock_redis):
          """Test complete eviction workflow."""
          # Set up scenario: high memory + large cache
          mock_redis.dbsize.return_value = 150

          with patch.object(cache_instance, '_get_memory_usage') as mock_memory, \
               patch.object(cache_instance, '_evict_expired_entries') as mock_expired, \
               patch.object(cache_instance, '_enforce_cache_limits') as mock_limits:

              mock_memory.return_value = 400.0  # High memory
              mock_expired.return_value = 5
              mock_limits.return_value = 15

              # Perform maintenance
              result = await cache_instance.perform_maintenance()

              # Verify workflow completed
              assert result["maintenance_performed"]
              assert result["evicted_expired"] == 5
              assert result["evicted_limits"] == 15
              assert result["memory_status"]["exceeds_warning"]
E       fixture 'cache_instance' not found
>       available fixtures: _session_event_loop, allocation_config, anyio_backend, anyio_backend_name, anyio_backend_options, assert_memory_usage, assert_performance_bounds, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, class_mocker, cleanup_test_artifacts, configure_test_logging, cov, doctest_namespace, ensemble_config, event_loop, event_loop_policy, generate_regime_data, generate_strategy_population, memory_monitor, mock_bot_engine_full, mock_bot_engine_simple, mock_component, mock_component_registry, mock_config_loader, mock_data_fetcher, mock_dataframe, mock_exchange, mock_failure_diagnosis, mock_heartbeat_message, mock_notifier, mock_order_manager, mock_order_result, mock_regime_detector, mock_regime_forecaster, mock_risk_manager, mock_self_healing_engine, mock_signal_router, mock_strategy, mock_strategy_generator, mock_strategy_selector, mock_timeframe_manager, mock_trading_signal, mocker, module_mocker, monkeypatch, multi_timeframe_data, no_cover, package_mocker, performance_timer, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, session_mocker, simulate_network_failure, synthetic_market_data, temp_dir, test_config, testrun_uid, tests/unit/test_cache_eviction.py::<event_loop>, tests/unit/test_cache_eviction.py::TestCacheIntegration::<event_loop>, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, worker_id
>       use 'pytest --fixtures [testpath]' for help on them.

C:\Users\TU\Desktop\new project\N1V1\tests\unit\test_cache_eviction.py:378
============================================================================== FAILURES ===============================================================================
________________________________________________________________________ test_data_validation _________________________________________________________________________
tests\data\test_data.py:130: in test_data_validation
    assert df.empty
E   assert False
E    +  where False =             open  high  low  close  volume\ntimestamp                                 \n2021-01-01   100   102   98    101    1000\n2021-01-02   101   103   99    102    1200\n2021-01-03   102   104  100    101     800.empty
___________________________________________________________________ test_historical_data_pagination ___________________________________________________________________
tests\data\test_data.py:305: in test_historical_data_pagination
    assert len(data) == 4  # Combined data from both pages
E   assert 3 == 4
E    +  where 3 = len(            close\n2023-01-01    100\n2023-01-02    101\n2023-01-03    103)
_________________________________________________________ TestDataFetcherInitialization.test_init_with_config _________________________________________________________
tests\data\test_data_fetcher.py:44: in test_init_with_config
    assert fetcher.cache_dir == '.test_cache'
E   AssertionError: assert 'C:\\Users\\T...\\.test_cache' == '.test_cache'
E
E     - .test_cache
E     + C:\Users\TU\Desktop\new project\N1V1\data\cache\.test_cache
__________________________________________________ TestDataFetcherInitialization.test_init_cache_directory_creation ___________________________________________________
tests\data\test_data_fetcher.py:87: in test_init_cache_directory_creation
    fetcher = DataFetcher(config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
__________________________________________________ TestDataFetcherAsyncMethods.test_get_historical_data_with_caching __________________________________________________
tests\data\test_data_fetcher.py:210: in test_get_historical_data_with_caching
    fetcher = DataFetcher(self.config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
____________________________________________ TestDataFetcherMultipleData.test_get_multiple_historical_data_partial_failure ____________________________________________
tests\data\test_data_fetcher.py:544: in test_get_multiple_historical_data_partial_failure
    assert 'ETH/USDT' not in result
E   AssertionError: assert 'ETH/USDT' not in {'BTC/USDT':                                                                       data\ntimestamp                     ...                                           \n2025-09-23 00:49:38.649  {'timestamp': '2022-01-01T00:00:00', 'open': 3...}
_______________________________________________ TestDataFetcherErrorScenarios.test_get_historical_data_unexpected_error _______________________________________________
tests\data\test_data_fetcher.py:630: in test_get_historical_data_unexpected_error
    assert result.empty
E   AssertionError: assert False
E    +  where False =                                                                       data\ntimestamp                                                                 \n2025-09-23 00:49:38.716  {'timestamp': '2022-01-01T00:00:00', 'open': 5....empty
_________________________________________________ TestDataFetcherErrorScenarios.test_cache_operations_error_handling __________________________________________________
tests\data\test_data_fetcher.py:676: in test_cache_operations_error_handling
    fetcher = DataFetcher(config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
__________________________________________________ TestDataFetcherEdgeCases.test_get_historical_data_empty_response ___________________________________________________
tests\data\test_data_fetcher.py:708: in test_get_historical_data_empty_response
    assert result.empty
E   AssertionError: assert False
E    +  where False =                                                                       data\ntimestamp                                                                 \n2025-09-23 00:49:38.716  {'timestamp': '2022-01-01T00:00:00', 'open': 5....empty
___________________________________________________ TestDataFetcherEdgeCases.test_get_historical_data_none_response ___________________________________________________
tests\data\test_data_fetcher.py:720: in test_get_historical_data_none_response
    assert result.empty
E   AssertionError: assert False
E    +  where False =                                                                       data\ntimestamp                                                                 \n2025-09-23 00:49:38.716  {'timestamp': '2022-01-01T00:00:00', 'open': 5....empty
__________________________________________ TestDataFetcherTimestampHandling.test_critical_cache_raises_exception_on_failure ___________________________________________
tests\data\test_data_fixes.py:363: in test_critical_cache_raises_exception_on_failure
    asyncio.run(self.fetcher._load_from_cache('btc_critical_key'))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got                                                                       data
E   timestamp
E   2025-09-23 03:03:13.425  {'timestamp': 'invalid', 'open': 100, 'high': ...
E   2025-09-23 03:03:13.425  {'timestamp': 'also_invalid', 'open': 101, 'hi...
E   2025-09-23 03:03:13.425  {'timestamp': 'still_invalid', 'open': 102, 'h...
_____________________________________________ TestDataFetcherTimestampHandling.test_timestamp_parsing_all_strategies_fail _____________________________________________
tests\data\test_data_fixes.py:331: in test_timestamp_parsing_all_strategies_fail
    result = asyncio.run(self.fetcher._load_from_cache('test_key'))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got                                                                       data
E   timestamp
E   2025-09-23 03:03:13.580  {'timestamp': 'invalid', 'open': 100, 'high': ...
E   2025-09-23 03:03:13.580  {'timestamp': 'also_invalid', 'open': 101, 'hi...
E   2025-09-23 03:03:13.580  {'timestamp': 'still_invalid', 'open': 102, 'h...
____________________________________________ TestDataFetcherTimestampHandling.test_timestamp_parsing_strategy_1_integer_ms ____________________________________________
tests\data\test_data_fixes.py:232: in test_timestamp_parsing_strategy_1_integer_ms
    result = asyncio.run(self.fetcher._load_from_cache('test_key'))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got                                                                       data
E   timestamp
E   2025-09-23 03:03:13.687  {'timestamp': 1640995200000, 'open': 100, 'hig...
E   2025-09-23 03:03:13.687  {'timestamp': 1640995260000, 'open': 101, 'hig...
E   2025-09-23 03:03:13.687  {'timestamp': 1640995320000, 'open': 102, 'hig...
_________________________________________ TestDataFetcherTimestampHandling.test_timestamp_parsing_strategy_2_datetime_column __________________________________________
tests\data\test_data_fixes.py:265: in test_timestamp_parsing_strategy_2_datetime_column
    result = asyncio.run(self.fetcher._load_from_cache('test_key'))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got                                                                       data
E   timestamp
E   2025-09-23 03:03:13.797  {'datetime': '2023-01-01 12:00:00', 'open': 10...
E   2025-09-23 03:03:13.797  {'datetime': '2023-01-01 12:01:00', 'open': 10...
E   2025-09-23 03:03:13.797  {'datetime': '2023-01-01 12:02:00', 'open': 10...
__________________________________________ TestDataFetcherTimestampHandling.test_timestamp_parsing_strategy_3_format_parsing __________________________________________
tests\data\test_data_fixes.py:298: in test_timestamp_parsing_strategy_3_format_parsing
    result = asyncio.run(self.fetcher._load_from_cache('test_key'))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got                                                                       data
E   timestamp
E   2025-09-23 03:03:13.883  {'timestamp': '2023-01-01T12:00:00', 'open': 1...
E   2025-09-23 03:03:13.883  {'timestamp': '2023-01-01T12:01:00', 'open': 1...
E   2025-09-23 03:03:13.883  {'timestamp': '2023-01-01T12:02:00', 'open': 1...
____________________________________________________ TestDataFetcherTimestampHandling.test_timezone_normalization _____________________________________________________
tests\data\test_data_fixes.py:392: in test_timezone_normalization
    result = asyncio.run(self.fetcher._load_from_cache('test_key'))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got                                                                       data
E   timestamp
E   2025-09-23 03:03:13.966  {'timestamp': '2023-01-01 00:00:00+00:00', 'op...
E   2025-09-23 03:03:13.966  {'timestamp': '2023-01-01 01:00:00+00:00', 'op...
E   2025-09-23 03:03:13.966  {'timestamp': '2023-01-01 02:00:00+00:00', 'op...
_________________________________________________________ TestDataOptimizations.test_cache_save_optimization __________________________________________________________
tests\data\test_data_fixes.py:774: in test_cache_save_optimization
    fetcher = DataFetcher(config)
data\data_fetcher.py:164: in __init__
    self._initialize_exchange()
data\data_fetcher.py:172: in _initialize_exchange
    sanitized_path = self._sanitize_cache_path(self.cache_dir)
data\data_fetcher.py:735: in _sanitize_cache_path
    raise PathTraversalError("Invalid cache directory path")
E   data.data_fetcher.PathTraversalError: Invalid cache directory path
____________________________________________ TestDatasetVersioningStructuredLogging.test_create_version_structured_logging ____________________________________________
tests\data\test_data_fixes.py:1292: in test_create_version_structured_logging
    version_id = self.version_manager.create_version(
data\dataset_versioning.py:263: in create_version
    raise DataValidationError("DataFrame must contain 'timestamp' column")
E   data.dataset_versioning.DataValidationError: DataFrame must contain 'timestamp' column
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:17 - data.dataset_versioning - WARNING - Metadata file not found: data\versions\version_metadata.json. Initializing with default metadata.
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.dataset_versioning:dataset_versioning.py:158 Metadata file not found: data\versions\version_metadata.json. Initializing with default metadata.
_______________________________________________________________ TestCaching.test_save_to_cache_success ________________________________________________________________
tests\data\test_data_module_refactoring.py:170: in test_save_to_cache_success
    result = asyncio.run(self.data_fetcher._save_to_cache(cache_key, df))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got None
______________________________________________________________ TestCaching.test_load_from_cache_success _______________________________________________________________
tests\data\test_data_module_refactoring.py:197: in test_load_from_cache_success
    asyncio.run(self.data_fetcher._save_to_cache(cache_key, df))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got None
____________________________________________________________ TestCaching.test_load_from_cache_nonexistent _____________________________________________________________
tests\data\test_data_module_refactoring.py:209: in test_load_from_cache_nonexistent
    result = asyncio.run(self.data_fetcher._load_from_cache('nonexistent_key'))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got None
______________________________________________________________ TestCaching.test_load_from_cache_expired _______________________________________________________________
tests\data\test_data_module_refactoring.py:232: in test_load_from_cache_expired
    result = asyncio.run(self.data_fetcher._load_from_cache(cache_key))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:37: in run
    raise ValueError("a coroutine was expected, got {!r}".format(main))
E   ValueError: a coroutine was expected, got                                                                       data
E   timestamp
E   2025-09-21 03:03:18.079  {'timestamp': '2023-01-01T00:00:00', 'open': 1...
_____________________________________________________________ TestVersioning.test_create_version_success ______________________________________________________________
tests\data\test_data_module_refactoring.py:415: in test_create_version_success
    result = self.version_manager.create_version(
data\dataset_versioning.py:263: in create_version
    raise DataValidationError("DataFrame must contain 'timestamp' column")
E   data.dataset_versioning.DataValidationError: DataFrame must contain 'timestamp' column
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-23 10:03:18 - data.dataset_versioning - WARNING - Metadata file not found: test_versions\version_metadata.json. Initializing with default metadata.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
WARNING  data.dataset_versioning:dataset_versioning.py:158 Metadata file not found: test_versions\version_metadata.json. Initializing with default metadata.
_____________________________________________________________ TestVersioning.test_migrate_legacy_dataset ______________________________________________________________
tests\data\test_data_module_refactoring.py:468: in test_migrate_legacy_dataset
    assert result is True
E   AssertionError: assert 'migrated_v2_634c96c3' is True
------------------------------------------------------------------------ Captured stdout setup ------------------------------------------------------------------------
2025-09-23 10:03:18 - data.dataset_versioning - WARNING - Metadata file not found: test_versions\version_metadata.json. Initializing with default metadata.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
WARNING  data.dataset_versioning:dataset_versioning.py:158 Metadata file not found: test_versions\version_metadata.json. Initializing with default metadata.
________________________________________________________ TestPathTraversal.test_sanitize_cache_path_traversal _________________________________________________________
tests\data\test_data_module_refactoring.py:493: in test_sanitize_cache_path_traversal
    with pytest.raises(PathTraversalError):
E   Failed: DID NOT RAISE <class 'data.data_fetcher.PathTraversalError'>
_________________________________________________ TestPathTraversalPrevention.test_sanitize_cache_path_traversal_dots _________________________________________________
tests\data\test_data_security.py:47: in test_sanitize_cache_path_traversal_dots
    fetcher._initialize_exchange()
data\data_fetcher.py:192: in _initialize_exchange
    exchange_class = getattr(ccxt, self.config['name'])
E   KeyError: 'name'
______________________________________________ TestPathTraversalPrevention.test_sanitize_cache_path_backslash_traversal _______________________________________________
tests\data\test_data_security.py:63: in test_sanitize_cache_path_backslash_traversal
    fetcher._initialize_exchange()
data\data_fetcher.py:192: in _initialize_exchange
    exchange_class = getattr(ccxt, self.config['name'])
E   KeyError: 'name'
_______________________________________________ TestPathTraversalPrevention.test_sanitize_cache_path_complex_traversal ________________________________________________
tests\data\test_data_security.py:71: in test_sanitize_cache_path_complex_traversal
    fetcher._initialize_exchange()
data\data_fetcher.py:192: in _initialize_exchange
    exchange_class = getattr(ccxt, self.config['name'])
E   KeyError: 'name'
__________________________________________________ TestPathTraversalPrevention.test_sanitize_cache_path_valid_nested __________________________________________________
tests\data\test_data_security.py:81: in test_sanitize_cache_path_valid_nested
    assert fetcher.cache_dir == expected_path
E   AssertionError: assert 'C:\\Users\\T...ted/cache/dir' == 'C:\\Users\\T...d\\cache\\dir'
E
E     Skipping 44 identical leading characters in diff, use -v to show
E     - che\nested\cache\dir
E     ?           ^     ^
E     + che\nested/cache/dir
E     ?           ^     ^
__________________________________________________ TestPathTraversalPrevention.test_sanitize_cache_path_empty_string __________________________________________________
tests\data\test_data_security.py:91: in test_sanitize_cache_path_empty_string
    assert fetcher.cache_dir == expected_path
E   AssertionError: assert 'C:\\Users\\T...data\\cache\\' == 'C:\\Users\\T...\\data\\cache'
E
E     Skipping 36 identical leading characters in diff, use -v to show
E     - \data\cache
E     + \data\cache\
E     ?            +
________________________________________________ TestDatasetVersionManagerSecurity.test_create_version_with_validation ________________________________________________
tests\data\test_data_security.py:248: in test_create_version_with_validation
    assert info['validation_passed'] is True
E   TypeError: 'NoneType' object is not subscriptable
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:19 - data.dataset_versioning - WARNING - Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmpe9qk1vaw\version_metadata.json. Initializing with default metadata.
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.dataset_versioning:dataset_versioning.py:158 Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmpe9qk1vaw\version_metadata.json. Initializing with default metadata.
______________________________________________ TestDatasetVersionManagerSecurity.test_migrate_legacy_dataset_validation _______________________________________________
tests\data\test_data_security.py:289: in test_migrate_legacy_dataset_validation
    assert version_id.startswith("migrated_v2_")
E   AttributeError: 'bool' object has no attribute 'startswith'
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:19 - data.dataset_versioning - WARNING - Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmprual0gvr\version_metadata.json. Initializing with default metadata.
2025-09-23 10:03:19 - root - INFO - Created binary labels: 5 trade signals out of 5 total samples
2025-09-23 10:03:19 - root - INFO - Binary label distribution: {1: 5}
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.dataset_versioning:dataset_versioning.py:158 Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmprual0gvr\version_metadata.json. Initializing with default metadata.
INFO     root:trainer.py:844 Created binary labels: 5 trade signals out of 5 total samples
INFO     root:trainer.py:845 Binary label distribution: {1: 5}
_____________________________________________ TestDatasetVersionManagerSecurity.test_migrate_legacy_dataset_invalid_input _____________________________________________
tests\data\test_data_security.py:310: in test_migrate_legacy_dataset_invalid_input
    with pytest.raises(DataValidationError):
E   Failed: DID NOT RAISE <class 'data.dataset_versioning.DataValidationError'>
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:19 - data.dataset_versioning - WARNING - Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmpn2ftxom_\version_metadata.json. Initializing with default metadata.
2025-09-23 10:03:19 - data.dataset_versioning - ERROR - Legacy dataset migration failed: DataFrame must have a 'timestamp' column or DatetimeIndex, duration=0.00s
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.dataset_versioning:dataset_versioning.py:158 Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmpn2ftxom_\version_metadata.json. Initializing with default metadata.
ERROR    data.dataset_versioning:dataset_versioning.py:621 Legacy dataset migration failed: DataFrame must have a 'timestamp' column or DatetimeIndex, duration=0.00s
_______________________________________________ TestConfigurationValidation.test_setup_data_directory_with_valid_config _______________________________________________
tests\data\test_data_security.py:604: in test_setup_data_directory_with_valid_config
    assert loader.data_dir.startswith(expected_base)
E   AssertionError: assert False
E    +  where False = <built-in method startswith of str object at 0x0000025868200AF0>('C:\\Users\\TU\\Desktop\\new project\\N1V1\\data\\historical')
E    +    where <built-in method startswith of str object at 0x0000025868200AF0> = 'historical_data'.startswith
E    +      where 'historical_data' = <data.historical_loader.HistoricalDataLoader object at 0x00000258031E5930>.data_dir
______________________________________________________ TestIntegrationSecurity.test_full_data_pipeline_security _______________________________________________________
tests\data\test_data_security.py:644: in test_full_data_pipeline_security
    assert loaded_df is not None
E   assert None is not None
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:19 - data.dataset_versioning - WARNING - Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmp3ffx73bu\version_metadata.json. Initializing with default metadata.
2025-09-23 10:03:19 - data.dataset_versioning - ERROR - Version integration_test_d97cea2a not found
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.dataset_versioning:dataset_versioning.py:158 Metadata file not found: C:\Users\TU\AppData\Local\Temp\tmp3ffx73bu\version_metadata.json. Initializing with default metadata.
ERROR    data.dataset_versioning:dataset_versioning.py:305 Version integration_test_d97cea2a not found
_______________________________________________________ TestIntegrationSecurity.test_error_handling_and_logging _______________________________________________________
tests\data\test_data_security.py:658: in test_error_handling_and_logging
    fetcher._initialize_exchange()
data\data_fetcher.py:192: in _initialize_exchange
    exchange_class = getattr(ccxt, self.config['name'])
E   KeyError: 'name'
__________________________________________________ TestHistoricalDataLoaderInitialization.test_setup_data_directory ___________________________________________________
tests\data\test_historical_loader.py:63: in test_setup_data_directory
    loader = HistoricalDataLoader(config, self.mock_data_fetcher)
data\historical_loader.py:63: in __init__
    self._setup_data_directory()
data\historical_loader.py:117: in _setup_data_directory
    self.data_dir = self._validate_data_directory(raw_data_dir)
data\historical_loader.py:86: in _validate_data_directory
    raise ConfigurationError("Absolute path detected")
E   data.historical_loader.ConfigurationError: Absolute path detected
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:19 - data.historical_loader - ERROR - Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmp4_bps4tz
2025-09-23 10:03:19 - data.historical_loader - ERROR - Data directory validation failed: Absolute path detected
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    data.historical_loader:historical_loader.py:85 Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmp4_bps4tz
ERROR    data.historical_loader:historical_loader.py:119 Data directory validation failed: Absolute path detected
________________________________________________ TestHistoricalDataLoaderAsyncMethods.test_load_symbol_data_from_cache ________________________________________________
tests\data\test_historical_loader.py:149: in test_load_symbol_data_from_cache
    loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)
data\historical_loader.py:63: in __init__
    self._setup_data_directory()
data\historical_loader.py:117: in _setup_data_directory
    self.data_dir = self._validate_data_directory(raw_data_dir)
data\historical_loader.py:86: in _validate_data_directory
    raise ConfigurationError("Absolute path detected")
E   data.historical_loader.ConfigurationError: Absolute path detected
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:20 - data.historical_loader - ERROR - Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmpfz9o8zr1
2025-09-23 10:03:20 - data.historical_loader - ERROR - Data directory validation failed: Absolute path detected
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    data.historical_loader:historical_loader.py:85 Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmpfz9o8zr1
ERROR    data.historical_loader:historical_loader.py:119 Data directory validation failed: Absolute path detected
______________________________________________ TestHistoricalDataLoaderAsyncMethods.test_load_symbol_data_force_refresh _______________________________________________
tests\data\test_historical_loader.py:182: in test_load_symbol_data_force_refresh
    loader = HistoricalDataLoader(self.config, self.mock_data_fetcher)
data\historical_loader.py:63: in __init__
    self._setup_data_directory()
data\historical_loader.py:117: in _setup_data_directory
    self.data_dir = self._validate_data_directory(raw_data_dir)
data\historical_loader.py:86: in _validate_data_directory
    raise ConfigurationError("Absolute path detected")
E   data.historical_loader.ConfigurationError: Absolute path detected
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:20 - data.historical_loader - ERROR - Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmp7zp6t2k6
2025-09-23 10:03:20 - data.historical_loader - ERROR - Data directory validation failed: Absolute path detected
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    data.historical_loader:historical_loader.py:85 Absolute path detected: C:\Users\TU\AppData\Local\Temp\tmp7zp6t2k6
ERROR    data.historical_loader:historical_loader.py:119 Data directory validation failed: Absolute path detected
_____________________________________________________ TestHistoricalDataLoaderDataProcessing.test_get_pandas_freq _____________________________________________________
tests\data\test_historical_loader.py:382: in test_get_pandas_freq
    assert result == expected
E   AssertionError: assert '1H' == '1h'
E
E     - 1h
E     + 1H
_____________________________________ TestHistoricalDataLoaderFetchCompleteHistory.test_fetch_complete_history_with_deduplication _____________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2237: in _execute_mock_call
    result = next(effect)
E   StopIteration

During handling of the above exception, another exception occurred:
tests\data\test_historical_loader.py:596: in test_fetch_complete_history_with_deduplication
    result = await self.loader._fetch_complete_history(
data\historical_loader.py:503: in _fetch_complete_history
    chunk = await self.data_fetcher.get_historical_data(symbol, timeframe, since=current_start)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2241: in _execute_mock_call
    raise StopAsyncIteration
E   StopAsyncIteration
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 1):
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 2):
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 3):
2025-09-23 10:03:21 - data.historical_loader - ERROR - Failed to fetch data for BTC/USDT after 3 attempts
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 1):
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 2):
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 3):
ERROR    data.historical_loader:historical_loader.py:512 Failed to fetch data for BTC/USDT after 3 attempts
________________________________________ TestHistoricalDataLoaderFetchCompleteHistory.test_fetch_complete_history_with_retries ________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2237: in _execute_mock_call
    result = next(effect)
E   StopIteration

During handling of the above exception, another exception occurred:
tests\data\test_historical_loader.py:626: in test_fetch_complete_history_with_retries
    result = await self.loader._fetch_complete_history(
data\historical_loader.py:503: in _fetch_complete_history
    chunk = await self.data_fetcher.get_historical_data(symbol, timeframe, since=current_start)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2241: in _execute_mock_call
    raise StopAsyncIteration
E   StopAsyncIteration
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 1): Network error
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 2): Network error
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 1):
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 2):
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 3):
2025-09-23 10:03:21 - data.historical_loader - ERROR - Failed to fetch data for BTC/USDT after 3 attempts
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 1): Network error
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 2): Network error
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 1):
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 2):
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 3):
ERROR    data.historical_loader:historical_loader.py:512 Failed to fetch data for BTC/USDT after 3 attempts
____________________________________ TestHistoricalDataLoaderFetchCompleteHistory.test_fetch_complete_history_max_retries_exceeded ____________________________________
tests\data\test_historical_loader.py:641: in test_fetch_complete_history_max_retries_exceeded
    result = await self.loader._fetch_complete_history(
data\historical_loader.py:503: in _fetch_complete_history
    chunk = await self.data_fetcher.get_historical_data(symbol, timeframe, since=current_start)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2234: in _execute_mock_call
    raise effect
data\historical_loader.py:503: in _fetch_complete_history
    chunk = await self.data_fetcher.get_historical_data(symbol, timeframe, since=current_start)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2234: in _execute_mock_call
    raise effect
data\historical_loader.py:503: in _fetch_complete_history
    chunk = await self.data_fetcher.get_historical_data(symbol, timeframe, since=current_start)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2234: in _execute_mock_call
    raise effect
E   Exception: Persistent error
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 1): Persistent error
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 2): Persistent error
2025-09-23 10:03:21 - data.historical_loader - WARNING - Error fetching data for BTC/USDT (attempt 3): Persistent error
2025-09-23 10:03:21 - data.historical_loader - ERROR - Failed to fetch data for BTC/USDT after 3 attempts
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 1): Persistent error
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 2): Persistent error
WARNING  data.historical_loader:historical_loader.py:510 Error fetching data for BTC/USDT (attempt 3): Persistent error
ERROR    data.historical_loader:historical_loader.py:512 Failed to fetch data for BTC/USDT after 3 attempts
__________________________________________________________________ TestMLFilter.test_save_load_model __________________________________________________________________
tests\ml\test_ml_filter.py:355: in test_save_load_model
    assert new_filter.model.is_trained
E   assert False
E    +  where False = <ml.ml_filter.LogisticRegressionModel object at 0x000002580546CEB0>.is_trained
E    +    where <ml.ml_filter.LogisticRegressionModel object at 0x000002580546CEB0> = <ml.ml_filter.MLFilter object at 0x000002580546D6F0>.model
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:50 - ml.ml_filter - INFO - LogisticRegression model trained on 16 samples
2025-09-23 10:03:50 - ml.ml_filter - INFO - Model validation metrics: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'mean_confidence': 0.9983558422279113}
2025-09-23 10:03:50 - ml.ml_filter - INFO - Model saved to C:\Users\TU\AppData\Local\Temp\tmph5lns66p.pkl
2025-09-23 10:03:50 - ml.ml_filter - INFO - ML Filter saved to C:\Users\TU\AppData\Local\Temp\tmph5lns66p.pkl
2025-09-23 10:03:50 - ml.model_loader - INFO - Loaded model from C:\Users\TU\AppData\Local\Temp\tmph5lns66p.pkl
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.ml_filter:ml_filter.py:181 LogisticRegression model trained on 16 samples
INFO     ml.ml_filter:ml_filter.py:491 Model validation metrics: {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'mean_confidence': 0.9983558422279113}
INFO     ml.ml_filter:ml_filter.py:142 Model saved to C:\Users\TU\AppData\Local\Temp\tmph5lns66p.pkl
INFO     ml.ml_filter:ml_filter.py:597 ML Filter saved to C:\Users\TU\AppData\Local\Temp\tmph5lns66p.pkl
INFO     ml.model_loader:model_loader.py:37 Loaded model from C:\Users\TU\AppData\Local\Temp\tmph5lns66p.pkl
______________________________________________________________ TestFactoryFunctions.test_load_ml_filter _______________________________________________________________
tests\ml\test_ml_filter.py:436: in test_load_ml_filter
    assert loaded_filter.model.is_trained
E   assert False
E    +  where False = <ml.ml_filter.LogisticRegressionModel object at 0x000002580540BC40>.is_trained
E    +    where <ml.ml_filter.LogisticRegressionModel object at 0x000002580540BC40> = <ml.ml_filter.MLFilter object at 0x000002580540AFB0>.model
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:51 - ml.ml_filter - INFO - LogisticRegression model trained on 8 samples
2025-09-23 10:03:51 - ml.ml_filter - INFO - Model saved to C:\Users\TU\AppData\Local\Temp\tmpoqy72raj.pkl
2025-09-23 10:03:51 - ml.ml_filter - INFO - ML Filter saved to C:\Users\TU\AppData\Local\Temp\tmpoqy72raj.pkl
2025-09-23 10:03:51 - ml.model_loader - INFO - Loaded model from C:\Users\TU\AppData\Local\Temp\tmpoqy72raj.pkl
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.ml_filter:ml_filter.py:181 LogisticRegression model trained on 8 samples
INFO     ml.ml_filter:ml_filter.py:142 Model saved to C:\Users\TU\AppData\Local\Temp\tmpoqy72raj.pkl
INFO     ml.ml_filter:ml_filter.py:597 ML Filter saved to C:\Users\TU\AppData\Local\Temp\tmpoqy72raj.pkl
INFO     ml.model_loader:model_loader.py:37 Loaded model from C:\Users\TU\AppData\Local\Temp\tmpoqy72raj.pkl
__________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_python_random __________________________________________________
tests\ml\test_reproducibility.py:39: in test_set_deterministic_seeds_python_random
    set_deterministic_seeds(42)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:03:54] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
______________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_numpy ______________________________________________________
tests\ml\test_reproducibility.py:54: in test_set_deterministic_seeds_numpy
    set_deterministic_seeds(42)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:03:54] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
_____________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_pandas ______________________________________________________
tests\ml\test_reproducibility.py:69: in test_set_deterministic_seeds_pandas
    set_deterministic_seeds(42)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:03:55] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
____________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_lightgbm _____________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'lgb'
_____________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_xgboost _____________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'xgb'
____________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_catboost _____________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'cb'
___________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_tensorflow ____________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'tf'
_____________________________________________________ TestDeterministicSeeds.test_set_deterministic_seeds_pytorch _____________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'torch'
_________________________________________________ TestEnvironmentSnapshot.test_capture_environment_snapshot_packages __________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'pkg_resources'
_________________________________________________ TestEnvironmentSnapshot.test_capture_environment_snapshot_env_vars __________________________________________________
tests\ml\test_reproducibility.py:187: in test_capture_environment_snapshot_env_vars
    assert env_snapshot['environment_variables']['PATH'] == '/usr/local/lib/python3.9'
E   AssertionError: assert '/usr/local/bin:/usr/bin' == '/usr/local/lib/python3.9'
E
E     - /usr/local/lib/python3.9
E     + /usr/local/bin:/usr/bin
_________________________________________________ TestEnvironmentSnapshot.test_capture_environment_snapshot_git_info __________________________________________________
tests\ml\test_reproducibility.py:209: in test_capture_environment_snapshot_git_info
    assert env_snapshot['git_info']['commit_hash'] == 'abc123def456'
E   KeyError: 'commit_hash'
________________________________________________ TestEnvironmentSnapshot.test_capture_environment_snapshot_git_failure ________________________________________________
tests\ml\test_reproducibility.py:222: in test_capture_environment_snapshot_git_failure
    assert 'error' in env_snapshot['git_info']
E   AssertionError: assert 'error' in {}
_________________________________________________ TestEnvironmentSnapshot.test_capture_environment_snapshot_hardware __________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'psutil'
_____________________________________________ TestEnvironmentSnapshot.test_capture_environment_snapshot_hardware_failure ______________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'psutil'
____________________________________________ TestExperimentTrackerReproducibility.test_experiment_tracker_git_info_logging ____________________________________________
tests\ml\test_reproducibility.py:354: in test_experiment_tracker_git_info_logging
    assert tracker.metadata['git_info']['commit_hash'] == 'def456abc789'
E   KeyError: 'commit_hash'
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:03:58 - ml.train - INFO - Experiment tracking initialized: test_experiment
2025-09-23 10:03:58 - ml.train - INFO - Experiment directory: C:\Users\TU\AppData\Local\Temp\tmp96mvhx0e\test_experiment
2025-09-23 10:03:58 - ml.train - INFO - Logged Git information for experiment test_experiment
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.train:train.py:602 Experiment tracking initialized: test_experiment
INFO     ml.train:train.py:603 Experiment directory: C:\Users\TU\AppData\Local\Temp\tmp96mvhx0e\test_experiment
INFO     ml.train:train.py:774 Logged Git information for experiment test_experiment
__________________________________________________ TestReproducibilityValidation.test_deterministic_numpy_operations __________________________________________________
tests\ml\test_reproducibility.py:390: in test_deterministic_numpy_operations
    set_deterministic_seeds(123)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:03:58] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
_________________________________________________ TestReproducibilityValidation.test_deterministic_pandas_operations __________________________________________________
tests\ml\test_reproducibility.py:405: in test_deterministic_pandas_operations
    set_deterministic_seeds(456)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:03:58] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
__________________________________________________ TestReproducibilityIntegration.test_full_reproducibility_workflow __________________________________________________
tests\ml\test_reproducibility.py:477: in test_full_reproducibility_workflow
    set_deterministic_seeds(seed)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:03:59] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
______________________________________________ TestReproducibilityIntegration.test_reproducibility_with_different_seeds _______________________________________________
tests\ml\test_reproducibility.py:518: in test_reproducibility_with_different_seeds
    set_deterministic_seeds(1)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:03:59] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
_________________________________________________ TestReproducibilityIntegration.test_reproducibility_with_same_seed __________________________________________________
tests\ml\test_reproducibility.py:531: in test_reproducibility_with_same_seed
    set_deterministic_seeds(100)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:04:00] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
___________________________________________________ TestPredictionProcessing.test_process_single_prediction_success ___________________________________________________
tests\ml\test_serving.py:216: in test_process_single_prediction_success
    assert "prediction" in result
E   TypeError: argument of type 'coroutine' is not iterable
_________________________________________________ TestPredictionProcessing.test_process_single_prediction_model_error _________________________________________________
tests\ml\test_serving.py:231: in test_process_single_prediction_model_error
    with pytest.raises(Exception):
E   Failed: DID NOT RAISE <class 'Exception'>
__________________________________________________________ TestPrepareTrainingData.test_prepare_data_success __________________________________________________________
venv\lib\site-packages\pandas\core\indexes\base.py:3812: in get_loc
    return self._engine.get_loc(casted_key)
pandas/_libs/index.pyx:167: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/index.pyx:196: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/hashtable_class_helper.pxi:7088: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
pandas/_libs/hashtable_class_helper.pxi:7096: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
E   KeyError: 'Open'

The above exception was the direct cause of the following exception:
tests\ml\test_train.py:133: in test_prepare_data_success
    result = prepare_training_data(data, min_samples=2)
ml\train.py:527: in prepare_training_data
    df = _process_symbol_data(df, outlier_threshold, column_map)
ml\train.py:393: in _process_symbol_data
    original_dtype = df[col].dtype
venv\lib\site-packages\pandas\core\frame.py:4107: in __getitem__
    indexer = self.columns.get_loc(key)
venv\lib\site-packages\pandas\core\indexes\base.py:3819: in get_loc
    raise KeyError(key) from err
E   KeyError: 'Open'
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:04:00 - ml.train - INFO - Preparing training data with comprehensive validation and parallel processing
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.train:train.py:471 Preparing training data with comprehensive validation and parallel processing
______________________________________________________ TestPrepareTrainingData.test_prepare_data_with_nan_values ______________________________________________________
venv\lib\site-packages\pandas\core\indexes\base.py:3812: in get_loc
    return self._engine.get_loc(casted_key)
pandas/_libs/index.pyx:167: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/index.pyx:196: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/hashtable_class_helper.pxi:7088: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
pandas/_libs/hashtable_class_helper.pxi:7096: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
E   KeyError: 'Open'

The above exception was the direct cause of the following exception:
tests\ml\test_train.py:163: in test_prepare_data_with_nan_values
    result = prepare_training_data(data, min_samples=2)
ml\train.py:527: in prepare_training_data
    df = _process_symbol_data(df, outlier_threshold, column_map)
ml\train.py:393: in _process_symbol_data
    original_dtype = df[col].dtype
venv\lib\site-packages\pandas\core\frame.py:4107: in __getitem__
    indexer = self.columns.get_loc(key)
venv\lib\site-packages\pandas\core\indexes\base.py:3819: in get_loc
    raise KeyError(key) from err
E   KeyError: 'Open'
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:04:01 - ml.train - INFO - Preparing training data with comprehensive validation and parallel processing
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.train:train.py:471 Preparing training data with comprehensive validation and parallel processing
_____________________________________________________ TestPrepareTrainingData.test_prepare_data_with_zero_prices ______________________________________________________
venv\lib\site-packages\pandas\core\indexes\base.py:3812: in get_loc
    return self._engine.get_loc(casted_key)
pandas/_libs/index.pyx:167: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/index.pyx:196: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/hashtable_class_helper.pxi:7088: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
pandas/_libs/hashtable_class_helper.pxi:7096: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
E   KeyError: 'Open'

The above exception was the direct cause of the following exception:
tests\ml\test_train.py:180: in test_prepare_data_with_zero_prices
    result = prepare_training_data(data, min_samples=2)
ml\train.py:527: in prepare_training_data
    df = _process_symbol_data(df, outlier_threshold, column_map)
ml\train.py:393: in _process_symbol_data
    original_dtype = df[col].dtype
venv\lib\site-packages\pandas\core\frame.py:4107: in __getitem__
    indexer = self.columns.get_loc(key)
venv\lib\site-packages\pandas\core\indexes\base.py:3819: in get_loc
    raise KeyError(key) from err
E   KeyError: 'Open'
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:04:01 - ml.train - INFO - Preparing training data with comprehensive validation and parallel processing
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.train:train.py:471 Preparing training data with comprehensive validation and parallel processing
______________________________________________________ TestPrepareTrainingData.test_prepare_data_outlier_removal ______________________________________________________
venv\lib\site-packages\pandas\core\indexes\base.py:3812: in get_loc
    return self._engine.get_loc(casted_key)
pandas/_libs/index.pyx:167: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/index.pyx:196: in pandas._libs.index.IndexEngine.get_loc
    ???
pandas/_libs/hashtable_class_helper.pxi:7088: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
pandas/_libs/hashtable_class_helper.pxi:7096: in pandas._libs.hashtable.PyObjectHashTable.get_item
    ???
E   KeyError: 'Open'

The above exception was the direct cause of the following exception:
tests\ml\test_train.py:198: in test_prepare_data_outlier_removal
    result = prepare_training_data(data, min_samples=10)
ml\train.py:527: in prepare_training_data
    df = _process_symbol_data(df, outlier_threshold, column_map)
ml\train.py:393: in _process_symbol_data
    original_dtype = df[col].dtype
venv\lib\site-packages\pandas\core\frame.py:4107: in __getitem__
    indexer = self.columns.get_loc(key)
venv\lib\site-packages\pandas\core\indexes\base.py:3819: in get_loc
    raise KeyError(key) from err
E   KeyError: 'Open'
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:04:02 - ml.train - INFO - Preparing training data with comprehensive validation and parallel processing
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.train:train.py:471 Preparing training data with comprehensive validation and parallel processing
_________________________________________________________________ TestMainFunction.test_main_success __________________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:890: in assert_not_called
    raise AssertionError(msg)
E   AssertionError: Expected 'exit' to not have been called. Called 1 times.
E   Calls: [call(1)].

During handling of the above exception, another exception occurred:
tests\ml\test_train.py:301: in test_main_success
    mock_exit.assert_not_called()
E   AssertionError: Expected 'exit' to not have been called. Called 1 times.
E   Calls: [call(1)].
E
E   pytest introspection follows:
E
E   Args:
E   assert (1,) == ()
E
E     Left contains one more item: 1
E     Use -v to get more diff
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:04:03 - ml.train - INFO - Loading configuration from config.json
2025-09-23 10:04:03 - ml.train - INFO - Experiment tracking initialized: train_all_20250923_100403
2025-09-23 10:04:03 - ml.train - INFO - Experiment directory: experiments\train_all_20250923_100403
2025-09-23 10:04:03 - ml.train - ERROR - Training failed: [10:04:03] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.train:train.py:958 Loading configuration from config.json
INFO     ml.train:train.py:602 Experiment tracking initialized: train_all_20250923_100403
INFO     ml.train:train.py:603 Experiment directory: experiments\train_all_20250923_100403
ERROR    ml.train:train.py:1107 Training failed: [10:04:03] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
________________________________________________________ TestMainFunction.test_main_predictive_models_disabled ________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:890: in assert_not_called
    raise AssertionError(msg)
E   AssertionError: Expected 'exit' to not have been called. Called 1 times.
E   Calls: [call(1)].

During handling of the above exception, another exception occurred:
tests\ml\test_train.py:334: in test_main_predictive_models_disabled
    mock_exit.assert_not_called()
E   AssertionError: Expected 'exit' to not have been called. Called 1 times.
E   Calls: [call(1)].
E
E   pytest introspection follows:
E
E   Args:
E   assert (1,) == ()
E
E     Left contains one more item: 1
E     Use -v to get more diff
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
2025-09-23 10:04:03 - ml.train - INFO - Loading configuration from config.json
2025-09-23 10:04:03 - ml.train - ERROR - Training failed: [WinError 123] The filename, directory name, or volume label syntax is incorrect: "experiments\\train_<MagicMock name='parse_args().symbol' id='2577059549760'>_20250923_100403"
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 962, in main
    tracker = initialize_experiment_tracking(args, config, seed=42)
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 895, in initialize_experiment_tracking
    tracker = ExperimentTracker(experiment_name=experiment_name)
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 582, in __init__
    self.experiment_dir.mkdir(parents=True, exist_ok=True)
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\pathlib.py", line 1175, in mkdir
    self._accessor.mkdir(self, mode)
OSError: [WinError 123] The filename, directory name, or volume label syntax is incorrect: "experiments\\train_<MagicMock name='parse_args().symbol' id='2577059549760'>_20250923_100403"
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     ml.train:train.py:958 Loading configuration from config.json
ERROR    ml.train:train.py:1107 Training failed: [WinError 123] The filename, directory name, or volume label syntax is incorrect: "experiments\\train_<MagicMock name='parse_args().symbol' id='2577059549760'>_20250923_100403"
_______________________________________________________ TestReproducibility.test_set_deterministic_seeds_basic ________________________________________________________
tests\ml\test_train.py:477: in test_set_deterministic_seeds_basic
    set_deterministic_seeds(seed=123)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:04:04] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
_____________________________________________________ TestReproducibility.test_set_deterministic_seeds_with_libs ______________________________________________________
tests\ml\test_train.py:491: in test_set_deterministic_seeds_with_libs
    set_deterministic_seeds(seed=456)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:04:04] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
__________________________________________________ TestReproducibility.test_set_deterministic_seeds_libs_unavailable __________________________________________________
tests\ml\test_train.py:503: in test_set_deterministic_seeds_libs_unavailable
    set_deterministic_seeds(seed=789)
ml\train.py:130: in set_deterministic_seeds
    xgb.set_config(seed=seed)
venv\lib\site-packages\xgboost\config.py:108: in wrap
    return func(*args, **kwargs)
venv\lib\site-packages\xgboost\config.py:132: in set_config
    _check_call(_LIB.XGBSetGlobalConfig(c_str(config)))
venv\lib\site-packages\xgboost\core.py:281: in _check_call
    raise XGBoostError(py_str(_LIB.XGBGetLastError()))
E   xgboost.core.XGBoostError: [10:04:04] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\c_api.cc:190: Unknown global parameters: { seed }
________________________________________________________ TestReproducibility.test_capture_environment_snapshot ________________________________________________________
tests\ml\test_train.py:527: in test_capture_environment_snapshot
    assert snapshot['platform'] == 'Linux-5.4.0'
E   AssertionError: assert 'x86_64' == 'Linux-5.4.0'
E
E     - Linux-5.4.0
E     + x86_64
________________________________________________ TestModelLoaderReproducibility.test_load_model_from_registry_success _________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1431: in __enter__
    self.target = self.getter()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1618: in <lambda>
    getter = lambda: _importer(target)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1257: in _importer
    thing = __import__(import_path)
E   ModuleNotFoundError: No module named 'mlflow'
________________________________________________ TestModelLoaderReproducibility.test_load_model_from_registry_fallback ________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1376: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1431: in __enter__
    self.target = self.getter()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1618: in <lambda>
    getter = lambda: _importer(target)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1257: in _importer
    thing = __import__(import_path)
E   ModuleNotFoundError: No module named 'mlflow'
______________________________________________________________ TestDataProcessing.test_load_data_success ______________________________________________________________
tests\ml\test_trainer.py:229: in test_load_data_success
    loaded_data = load_data(f.name)
ml\trainer.py:239: in load_data
    logger.info(f"Loading data from {path}")
E   NameError: name 'logger' is not defined
__________________________________________________________ TestDataProcessing.test_load_data_file_not_found ___________________________________________________________
tests\ml\test_trainer.py:240: in test_load_data_file_not_found
    load_data("non_existent_file.csv")
ml\trainer.py:239: in load_data
    logger.info(f"Loading data from {path}")
E   NameError: name 'logger' is not defined
______________________________________________________ TestAssetSelector.test_fetch_from_coingecko_async_success ______________________________________________________
tests\optimization\test_asset_selector.py:398: in test_fetch_from_coingecko_async_success
    assert result == {'ETH/USDT': 50000000000}
E   AssertionError: assert None == {'ETH/USDT': 50000000000}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     AssetSelector:asset_selector.py:142 AssetSelector initialized with 3 available assets
WARNING  AssetSelector:asset_selector.py:529 Failed to fetch from CoinGecko: __aenter__
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestAssetSelector.test_apply_market_cap_weighting __________________________________________________________
tests\optimization\test_cross_asset_validation.py:387: in test_apply_market_cap_weighting
    assert eth_weight > ada_weight
E   assert 0.5 > 0.5
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
WARNING  AssetSelector:cross_asset_validation.py:448 Failed to fetch from CoinGecko: Invalid variable type: value should be str, int or float, got False of type <class 'bool'>
WARNING  AssetSelector:cross_asset_validation.py:359 No market cap data available, falling back to equal weights
__________________________________________________ TestWalkForwardOptimizer.test_optimization_with_insufficient_data __________________________________________________
tests\optimization\test_optimization.py:124: in test_optimization_with_insufficient_data
    result = optimizer.optimize(mock_strategy, data)
optimization\walk_forward.py:622: in optimize
    return self._optimize_memory_efficient(strategy_class, data)
optimization\walk_forward.py:733: in _optimize_memory_efficient
    total_time = time.time() - start_time
E   NameError: name 'start_time' is not defined
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:55,287 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:55,287 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:55,287 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:55,287 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:55,289 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:55,289 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:55,289 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:55,289 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:55,290 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:55,291 - WalkForwardOptimizer - INFO - Total windows created: 0
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     WalkForwardOptimizer:walk_forward.py:401 Walk-Forward Optimizer initialized
INFO     WalkForwardOptimizer:walk_forward.py:617 Starting Walk-Forward Optimization
INFO     WalkForwardOptimizer:walk_forward.py:621 Using memory-efficient processing mode
INFO     WalkForwardOptimizer:walk_forward.py:675 Processing data in chunks of 50000 rows
INFO     WalkForwardOptimizer:walk_forward.py:685 Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
INFO     WalkForwardOptimizer:walk_forward.py:692 Processing chunk 1/1
INFO     WalkForwardOptimizer:walk_forward.py:693 Chunk shape: (5, 1)
WARNING  WalkForwardDataSplitter:walk_forward.py:215 Insufficient data: 5 samples, minimum 50
INFO     WalkForwardOptimizer:walk_forward.py:721 Total windows created: 0
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________________ TestGeneticOptimizer.test_population_initialization _________________________________________________________
tests\optimization\test_optimization.py:191: in test_population_initialization
    assert len(optimizer.population) == 10
E   AssertionError: assert 20 == 10
E    +  where 20 = len([Chromosome(genes={'rsi_period': 47}, fitness=-inf), Chromosome(genes={'rsi_period': 31}, fitness=-inf), Chromosome(ge...ess=-inf), Chromosome(genes={'rsi_period': 38}, fitness=-inf), Chromosome(genes={'rsi_period': 33}, fitness=-inf), ...])
E    +    where [Chromosome(genes={'rsi_period': 47}, fitness=-inf), Chromosome(genes={'rsi_period': 31}, fitness=-inf), Chromosome(ge...ess=-inf), Chromosome(genes={'rsi_period': 38}, fitness=-inf), Chromosome(genes={'rsi_period': 33}, fitness=-inf), ...] = <optimization.genetic_optimizer.GeneticOptimizer object at 0x0000025806AEA680>.population
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:55,380 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes
2025-09-23 10:04:55,380 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes
2025-09-23 10:04:55,380 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes
2025-09-23 10:04:55,380 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     GeneticOptimizer:genetic_optimizer.py:460 Initialized population with 20 chromosomes
____________________________________________________ TestGeneticOptimizer.test_population_initialization_no_bounds ____________________________________________________
tests\optimization\test_optimization.py:203: in test_population_initialization_no_bounds
    assert len(optimizer.population) == 5
E   assert 20 == 5
E    +  where 20 = len([Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), ...])
E    +    where [Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), Chromosome(genes={}, fitness=-inf), ...] = <optimization.genetic_optimizer.GeneticOptimizer object at 0x00000258067CD450>.population
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:55,415 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes (empty genes)
2025-09-23 10:04:55,415 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes (empty genes)
2025-09-23 10:04:55,415 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes (empty genes)
2025-09-23 10:04:55,415 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes (empty genes)
2025-09-23 10:04:55,415 - GeneticOptimizer - INFO - Initialized population with 20 chromosomes (empty genes)
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     GeneticOptimizer:genetic_optimizer.py:441 Initialized population with 20 chromosomes (empty genes)
______________________________________________ TestCoreOptimizationAlgorithms.test_crossover_with_different_gene_counts _______________________________________________
tests\optimization\test_optimization.py:720: in test_crossover_with_different_gene_counts
    offspring1, offspring2 = optimizer._crossover(parent1, parent2)
optimization\genetic_optimizer.py:549: in _crossover
    genes1[param], genes2[param] = genes2[param], genes1[param]
E   KeyError: 'param2'
_________________________________________________ TestCoreOptimizationAlgorithms.test_parameter_validation_edge_cases _________________________________________________
tests\optimization\test_optimization.py:903: in test_parameter_validation_edge_cases
    assert not optimizer.validate_parameters({'extreme_param': -2000})
E   AssertionError: assert not True
E    +  where True = <bound method BaseOptimizer.validate_parameters of <optimization.genetic_optimizer.GeneticOptimizer object at 0x0000025806B854B0>>({'extreme_param': -2000})
E    +    where <bound method BaseOptimizer.validate_parameters of <optimization.genetic_optimizer.GeneticOptimizer object at 0x0000025806B854B0>> = <optimization.genetic_optimizer.GeneticOptimizer object at 0x0000025806B854B0>.validate_parameters
________________________________________________________________ TestRLOptimizer.test_policy_save_load ________________________________________________________________
tests\optimization\test_optimization.py:1123: in test_policy_save_load
    assert len(new_optimizer.q_table) > 0
E   assert 0 > 0
E    +  where 0 = len(defaultdict(<function RLOptimizer.__init__.<locals>.<lambda> at 0x0000025806E1FEB0>, {}))
E    +    where defaultdict(<function RLOptimizer.__init__.<locals>.<lambda> at 0x0000025806E1FEB0>, {}) = <optimization.rl_optimizer.RLOptimizer object at 0x0000025806B57A90>.q_table
_____________________________________________________________ TestOptimizerFactory.test_create_optimizer ______________________________________________________________
tests\optimization\test_optimization.py:1140: in test_create_optimizer
    assert optimizer.population_size == 10
E   assert 20 == 10
E    +  where 20 = <optimization.genetic_optimizer.GeneticOptimizer object at 0x00000258072C2680>.population_size
________________________________________________ TestCrossPairValidation.test_cross_pair_validation_insufficient_data _________________________________________________
tests\optimization\test_optimization.py:1414: in test_cross_pair_validation_insufficient_data
    results = optimizer.cross_pair_validation(
optimization\walk_forward.py:496: in cross_pair_validation
    best_params = self.optimize(strategy, train_data)
optimization\walk_forward.py:622: in optimize
    return self._optimize_memory_efficient(strategy_class, data)
optimization\walk_forward.py:733: in _optimize_memory_efficient
    total_time = time.time() - start_time
E   NameError: name 'start_time' is not defined
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,519 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,521 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,522 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,523 - WalkForwardOptimizer - INFO - Chunk shape: (50, 1)
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,524 - WalkForwardOptimizer - INFO - Total windows created: 0
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     WalkForwardOptimizer:walk_forward.py:401 Walk-Forward Optimizer initialized
INFO     WalkForwardOptimizer:walk_forward.py:617 Starting Walk-Forward Optimization
INFO     WalkForwardOptimizer:walk_forward.py:621 Using memory-efficient processing mode
INFO     WalkForwardOptimizer:walk_forward.py:675 Processing data in chunks of 50000 rows
INFO     WalkForwardOptimizer:walk_forward.py:685 Data chunking info: {'total_rows': 50, 'chunk_size': 50000, 'overlap': 25, 'estimated_chunks': 1, 'memory_efficient': True}
INFO     WalkForwardOptimizer:walk_forward.py:692 Processing chunk 1/1
INFO     WalkForwardOptimizer:walk_forward.py:693 Chunk shape: (50, 1)
INFO     WalkForwardDataSplitter:walk_forward.py:300 Created 0 walk-forward windows
INFO     WalkForwardOptimizer:walk_forward.py:721 Total windows created: 0
__________________________________________________________ TestWalkForwardOptimizer.test_optimize_no_windows __________________________________________________________
tests\optimization\test_walk_forward.py:272: in test_optimize_no_windows
    result = optimizer.optimize(MockStrategy, pd.DataFrame())
optimization\walk_forward.py:622: in optimize
    return self._optimize_memory_efficient(strategy_class, data)
optimization\walk_forward.py:733: in _optimize_memory_efficient
    total_time = time.time() - start_time
E   NameError: name 'start_time' is not defined
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,723 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,724 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,725 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,727 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,729 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:56,730 - WalkForwardOptimizer - INFO - Total windows created: 0
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     WalkForwardOptimizer:walk_forward.py:401 Walk-Forward Optimizer initialized
INFO     WalkForwardOptimizer:walk_forward.py:617 Starting Walk-Forward Optimization
INFO     WalkForwardOptimizer:walk_forward.py:621 Using memory-efficient processing mode
INFO     WalkForwardOptimizer:walk_forward.py:675 Processing data in chunks of 50000 rows
INFO     WalkForwardOptimizer:walk_forward.py:685 Data chunking info: {'total_rows': 0, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 0, 'memory_efficient': True}
INFO     WalkForwardOptimizer:walk_forward.py:721 Total windows created: 0
_________________________________________________________ TestWalkForwardOptimizer.test_optimize_with_windows _________________________________________________________
tests\optimization\test_walk_forward.py:303: in test_optimize_with_windows
    result = optimizer.optimize(MockStrategy, sample_data)
optimization\walk_forward.py:622: in optimize
    return self._optimize_memory_efficient(strategy_class, data)
optimization\walk_forward.py:710: in _optimize_memory_efficient
    self._process_windows_sequential(strategy_class, chunk_windows)
E   TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,801 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,803 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,804 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,805 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,806 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
2025-09-23 10:04:56,809 - WalkForwardOptimizer - INFO - Created 7 windows from chunk 1
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     WalkForwardOptimizer:walk_forward.py:401 Walk-Forward Optimizer initialized
INFO     WalkForwardOptimizer:walk_forward.py:617 Starting Walk-Forward Optimization
INFO     WalkForwardOptimizer:walk_forward.py:621 Using memory-efficient processing mode
INFO     WalkForwardOptimizer:walk_forward.py:675 Processing data in chunks of 50000 rows
INFO     WalkForwardOptimizer:walk_forward.py:685 Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 15, 'estimated_chunks': 1, 'memory_efficient': True}
INFO     WalkForwardOptimizer:walk_forward.py:692 Processing chunk 1/1
INFO     WalkForwardOptimizer:walk_forward.py:693 Chunk shape: (200, 5)
INFO     WalkForwardDataSplitter:walk_forward.py:300 Created 7 walk-forward windows
INFO     WalkForwardOptimizer:walk_forward.py:704 Created 7 windows from chunk 1
___________________________________________________________ TestIntegration.test_full_walk_forward_workflow ___________________________________________________________
tests\optimization\test_walk_forward.py:605: in test_full_walk_forward_workflow
    best_params = optimizer.optimize(MockStrategy, sample_data)
optimization\walk_forward.py:622: in optimize
    return self._optimize_memory_efficient(strategy_class, data)
optimization\walk_forward.py:710: in _optimize_memory_efficient
    self._process_windows_sequential(strategy_class, chunk_windows)
E   TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,989 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,991 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,992 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,993 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,995 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:56,997 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
2025-09-23 10:04:57,001 - WalkForwardOptimizer - INFO - Created 10 windows from chunk 1
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     WalkForwardOptimizer:walk_forward.py:401 Walk-Forward Optimizer initialized
INFO     WalkForwardOptimizer:walk_forward.py:617 Starting Walk-Forward Optimization
INFO     WalkForwardOptimizer:walk_forward.py:621 Using memory-efficient processing mode
INFO     WalkForwardOptimizer:walk_forward.py:675 Processing data in chunks of 50000 rows
INFO     WalkForwardOptimizer:walk_forward.py:685 Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 12, 'estimated_chunks': 1, 'memory_efficient': True}
INFO     WalkForwardOptimizer:walk_forward.py:692 Processing chunk 1/1
INFO     WalkForwardOptimizer:walk_forward.py:693 Chunk shape: (200, 5)
INFO     WalkForwardDataSplitter:walk_forward.py:300 Created 10 walk-forward windows
INFO     WalkForwardOptimizer:walk_forward.py:704 Created 10 windows from chunk 1
________________________________________________________ TestIntegration.test_error_handling_insufficient_data ________________________________________________________
tests\optimization\test_walk_forward.py:629: in test_error_handling_insufficient_data
    result = optimizer.optimize(MockStrategy, small_data)
optimization\walk_forward.py:622: in optimize
    return self._optimize_memory_efficient(strategy_class, data)
optimization\walk_forward.py:733: in _optimize_memory_efficient
    total_time = time.time() - start_time
E   NameError: name 'start_time' is not defined
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,060 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,065 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,066 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,067 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,068 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,069 - WalkForwardOptimizer - INFO - Chunk shape: (5, 1)
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
2025-09-23 10:04:57,070 - WalkForwardOptimizer - INFO - Total windows created: 0
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     WalkForwardOptimizer:walk_forward.py:401 Walk-Forward Optimizer initialized
INFO     WalkForwardOptimizer:walk_forward.py:617 Starting Walk-Forward Optimization
INFO     WalkForwardOptimizer:walk_forward.py:621 Using memory-efficient processing mode
INFO     WalkForwardOptimizer:walk_forward.py:675 Processing data in chunks of 50000 rows
INFO     WalkForwardOptimizer:walk_forward.py:685 Data chunking info: {'total_rows': 5, 'chunk_size': 50000, 'overlap': 50, 'estimated_chunks': 1, 'memory_efficient': True}
INFO     WalkForwardOptimizer:walk_forward.py:692 Processing chunk 1/1
INFO     WalkForwardOptimizer:walk_forward.py:693 Chunk shape: (5, 1)
WARNING  WalkForwardDataSplitter:walk_forward.py:215 Insufficient data: 5 samples, minimum 100
INFO     WalkForwardOptimizer:walk_forward.py:721 Total windows created: 0
________________________________________________________ TestIntegration.test_error_handling_optimizer_failure ________________________________________________________
tests\optimization\test_walk_forward.py:652: in test_error_handling_optimizer_failure
    result = optimizer.optimize(MockStrategy, sample_data)
optimization\walk_forward.py:622: in optimize
    return self._optimize_memory_efficient(strategy_class, data)
optimization\walk_forward.py:710: in _optimize_memory_efficient
    self._process_windows_sequential(strategy_class, chunk_windows)
E   TypeError: WalkForwardOptimizer._process_windows_sequential() takes 2 positional arguments but 3 were given
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,635 - WalkForwardOptimizer - INFO - Walk-Forward Optimizer initialized
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,637 - WalkForwardOptimizer - INFO - Starting Walk-Forward Optimization
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,638 - WalkForwardOptimizer - INFO - Using memory-efficient processing mode
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,639 - WalkForwardOptimizer - INFO - Processing data in chunks of 50000 rows
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,640 - WalkForwardOptimizer - INFO - Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,641 - WalkForwardOptimizer - INFO - Processing chunk 1/1
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,642 - WalkForwardOptimizer - INFO - Chunk shape: (200, 5)
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
2025-09-23 10:04:57,649 - WalkForwardOptimizer - INFO - Created 9 windows from chunk 1
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     WalkForwardOptimizer:walk_forward.py:401 Walk-Forward Optimizer initialized
INFO     WalkForwardOptimizer:walk_forward.py:617 Starting Walk-Forward Optimization
INFO     WalkForwardOptimizer:walk_forward.py:621 Using memory-efficient processing mode
INFO     WalkForwardOptimizer:walk_forward.py:675 Processing data in chunks of 50000 rows
INFO     WalkForwardOptimizer:walk_forward.py:685 Data chunking info: {'total_rows': 200, 'chunk_size': 50000, 'overlap': 10, 'estimated_chunks': 1, 'memory_efficient': True}
INFO     WalkForwardOptimizer:walk_forward.py:692 Processing chunk 1/1
INFO     WalkForwardOptimizer:walk_forward.py:693 Chunk shape: (200, 5)
INFO     WalkForwardDataSplitter:walk_forward.py:300 Created 9 walk-forward windows
INFO     WalkForwardOptimizer:walk_forward.py:704 Created 9 windows from chunk 1
___________________________________________________ TestMarketConditionMonitor.test_assess_market_conditions_normal ___________________________________________________
risk\adaptive_policy.py:120: in assess_market_conditions
    self._validate_market_data(market_data)
risk\adaptive_policy.py:480: in _validate_market_data
    raise ValueError("Market data validation failed - missing required columns or invalid data")
E   ValueError: Market data validation failed - missing required columns or invalid data

The above exception was the direct cause of the following exception:
tests\risk\test_adaptive_policy.py:60: in test_assess_market_conditions_normal
    conditions = monitor.assess_market_conditions('BTC/USDT', data)
risk\adaptive_policy.py:140: in assess_market_conditions
    raise ValueError(f"Critical market data validation error for {symbol}: {e}") from e
E   ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required columns or invalid data
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    risk.adaptive_policy:adaptive_policy.py:136 Error assessing market conditions for BTC/USDT: Market data validation failed - missing required columns or invalid data
ERROR    risk.adaptive_policy:adaptive_policy.py:137 Stack trace: Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\risk\adaptive_policy.py", line 120, in assess_market_conditions
    self._validate_market_data(market_data)
  File "C:\Users\TU\Desktop\new project\N1V1\risk\adaptive_policy.py", line 480, in _validate_market_data
    raise ValueError("Market data validation failed - missing required columns or invalid data")
ValueError: Market data validation failed - missing required columns or invalid data
______________________________________________ TestMarketConditionMonitor.test_assess_market_conditions_high_volatility _______________________________________________
risk\adaptive_policy.py:120: in assess_market_conditions
    self._validate_market_data(market_data)
risk\adaptive_policy.py:480: in _validate_market_data
    raise ValueError("Market data validation failed - missing required columns or invalid data")
E   ValueError: Market data validation failed - missing required columns or invalid data

The above exception was the direct cause of the following exception:
tests\risk\test_adaptive_policy.py:85: in test_assess_market_conditions_high_volatility
    conditions = monitor.assess_market_conditions('BTC/USDT', data)
risk\adaptive_policy.py:140: in assess_market_conditions
    raise ValueError(f"Critical market data validation error for {symbol}: {e}") from e
E   ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required columns or invalid data
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    risk.adaptive_policy:adaptive_policy.py:136 Error assessing market conditions for BTC/USDT: Market data validation failed - missing required columns or invalid data
ERROR    risk.adaptive_policy:adaptive_policy.py:137 Stack trace: Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\risk\adaptive_policy.py", line 120, in assess_market_conditions
    self._validate_market_data(market_data)
  File "C:\Users\TU\Desktop\new project\N1V1\risk\adaptive_policy.py", line 480, in _validate_market_data
    raise ValueError("Market data validation failed - missing required columns or invalid data")
ValueError: Market data validation failed - missing required columns or invalid data
________________________________________________________________ TestEdgeCases.test_empty_market_data _________________________________________________________________
tests\risk\test_adaptive_policy.py:507: in test_empty_market_data
    assert multiplier == 1.0
E   assert 0.25 == 1.0
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.adaptive_policy:adaptive_policy.py:691 AdaptiveRiskPolicy initialized
WARNING  risk.adaptive_policy:adaptive_policy.py:797 Insufficient market data for BTC/USDT, using conservative fallback
WARNING  risk.adaptive_policy:adaptive_policy.py:1125 Using conservative fallback for insufficient data: multiplier=0.25
_________________________________________________________________ TestEdgeCases.test_none_market_data _________________________________________________________________
tests\risk\test_adaptive_policy.py:518: in test_none_market_data
    assert multiplier == 1.0
E   assert 0.25 == 1.0
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.adaptive_policy:adaptive_policy.py:691 AdaptiveRiskPolicy initialized
WARNING  risk.adaptive_policy:adaptive_policy.py:797 Insufficient market data for BTC/USDT, using conservative fallback
WARNING  risk.adaptive_policy:adaptive_policy.py:1125 Using conservative fallback for insufficient data: multiplier=0.25
__________________________________________________________ TestEdgeCases.test_market_monitor_error_handling ___________________________________________________________
risk\adaptive_policy.py:120: in assess_market_conditions
    self._validate_market_data(market_data)
risk\adaptive_policy.py:480: in _validate_market_data
    raise ValueError("Market data validation failed - missing required columns or invalid data")
E   ValueError: Market data validation failed - missing required columns or invalid data

The above exception was the direct cause of the following exception:
tests\risk\test_adaptive_policy.py:566: in test_market_monitor_error_handling
    conditions = monitor.assess_market_conditions('BTC/USDT', data)
risk\adaptive_policy.py:140: in assess_market_conditions
    raise ValueError(f"Critical market data validation error for {symbol}: {e}") from e
E   ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required columns or invalid data
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    risk.adaptive_policy:adaptive_policy.py:136 Error assessing market conditions for BTC/USDT: Market data validation failed - missing required columns or invalid data
ERROR    risk.adaptive_policy:adaptive_policy.py:137 Stack trace: Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\risk\adaptive_policy.py", line 120, in assess_market_conditions
    self._validate_market_data(market_data)
  File "C:\Users\TU\Desktop\new project\N1V1\risk\adaptive_policy.py", line 480, in _validate_market_data
    raise ValueError("Market data validation failed - missing required columns or invalid data")
ValueError: Market data validation failed - missing required columns or invalid data
_________________________________________________ TestAdaptiveRiskManagement.test_adaptive_position_size_calculation __________________________________________________
tests\risk\test_adaptive_risk.py:141: in test_adaptive_position_size_calculation
    assert isinstance(position_size, Decimal)
E   assert False
E    +  where False = isinstance(<coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x0000025806763B50>, Decimal)
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________ TestAdaptiveRiskManagement.test_adaptive_position_size_fallback ___________________________________________________
tests\risk\test_adaptive_risk.py:168: in test_adaptive_position_size_fallback
    assert position_size == expected
E   AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x0000025806995FC0> == Decimal('200.00')
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________ TestAdaptiveRiskManagement.test_position_sizing_method_selection ___________________________________________________
tests\risk\test_adaptive_risk.py:482: in test_position_sizing_method_selection
    size1 = await risk_manager.calculate_position_size(signal, sample_market_data)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________ TestAdaptiveRiskManagement.test_error_handling_in_calculations ____________________________________________________
tests\risk\test_adaptive_risk.py:519: in test_error_handling_in_calculations
    assert pos_size >= 0
E   TypeError: '>=' not supported between instances of 'coroutine' and 'int'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________________ TestAdaptiveRiskIntegration.test_complete_trade_workflow _______________________________________________________
tests\risk\test_adaptive_risk.py:660: in test_complete_trade_workflow
    assert is_valid is True
E   assert False is True
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:04:58.910685Z", "level": "TRADE", "module": "crypto_bot", "message": "Signal rejected: error: [<class 'decimal.ConversionSyntax'>]", "correlation_id": null, "request_id": null, "strategy_id": null}
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:04:58.904000+00:00) to int (1758596698904)
INFO     risk.anomaly_detector:anomaly_detector.py:442 Using conservative anomaly result: insufficient data
ERROR    risk.risk_manager:risk_manager.py:368 Error evaluating signal: [<class 'decimal.ConversionSyntax'>]
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\risk\risk_manager.py", line 325, in evaluate_signal
    signal.amount = await self.calculate_position_size(signal, market_data)
  File "C:\Users\TU\Desktop\new project\N1V1\risk\risk_manager.py", line 418, in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:04:58.904000+00:00) to int (1758596698904)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestPriceGapDetector.test_detect_normal_price_change _________________________________________________________
tests\risk\test_anomaly_detector.py:182: in test_detect_normal_price_change
    assert result.is_anomaly is False
E   AssertionError: assert False is False
E    +  where False = AnomalyResult(is_anomaly=False, anomaly_type=<AnomalyType.NONE: 'none'>, severity=<AnomalySeverity.LOW: 'low'>, confid...0, 'current_close': 102.0, 'gap_pct': 0.9900990099009901}, timestamp=datetime.datetime(2025, 9, 23, 10, 4, 58, 993900)).is_anomaly
_____________________________________________________________ TestPriceGapDetector.test_detect_price_gap ______________________________________________________________
tests\risk\test_anomaly_detector.py:194: in test_detect_price_gap
    assert result.is_anomaly is True
E   AssertionError: assert True is True
E    +  where True = AnomalyResult(is_anomaly=True, anomaly_type=<AnomalyType.PRICE_GAP: 'price_gap'>, severity=<AnomalySeverity.CRITICAL: ...ev_close': 100.0, 'current_close': 115.0, 'gap_pct': 15.0}, timestamp=datetime.datetime(2025, 9, 23, 10, 4, 59, 16267)).is_anomaly
_____________________________________________________________ TestPriceGapDetector.test_detect_small_gap ______________________________________________________________
tests\risk\test_anomaly_detector.py:208: in test_detect_small_gap
    assert result.is_anomaly is False
E   AssertionError: assert False is False
E    +  where False = AnomalyResult(is_anomaly=False, anomaly_type=<AnomalyType.NONE: 'none'>, severity=<AnomalySeverity.LOW: 'low'>, confid...rev_close': 100.0, 'current_close': 102.0, 'gap_pct': 2.0}, timestamp=datetime.datetime(2025, 9, 23, 10, 4, 59, 40073)).is_anomaly
_________________________________________________________ TestAnomalyDetector.test_initialization_with_config _________________________________________________________
tests\risk\test_anomaly_detector.py:235: in test_initialization_with_config
    detector = AnomalyDetector(config)
risk\anomaly_detector.py:874: in __init__
    self.scale_down_threshold = self._string_to_severity(self.response_config['scale_down_threshold'])
E   KeyError: 'scale_down_threshold'
___________________________________________________________ TestAnomalyDetector.test_get_anomaly_statistics ___________________________________________________________
tests\risk\test_anomaly_detector.py:329: in test_get_anomaly_statistics
    stats = detector.get_anomaly_statistics()
E   AttributeError: 'AnomalyDetector' object has no attribute 'get_anomaly_statistics'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
______________________________________________________________ TestAnomalyDetector.test_empty_statistics ______________________________________________________________
tests\risk\test_anomaly_detector.py:339: in test_empty_statistics
    stats = detector.get_anomaly_statistics()
E   AttributeError: 'AnomalyDetector' object has no attribute 'get_anomaly_statistics'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
_________________________________________________________________ TestAnomalyLogging.test_log_to_file _________________________________________________________________
tests\risk\test_anomaly_detector.py:356: in test_log_to_file
    detector = AnomalyDetector({
risk\anomaly_detector.py:878: in __init__
    self.log_anomalies = self.config['logging']['enabled']
E   KeyError: 'enabled'
_________________________________________________________________ TestAnomalyLogging.test_log_to_json _________________________________________________________________
tests\risk\test_anomaly_detector.py:393: in test_log_to_json
    detector = AnomalyDetector({
risk\anomaly_detector.py:878: in __init__
    self.log_anomalies = self.config['logging']['enabled']
E   KeyError: 'enabled'
__________________________________________________________ TestAnomalyLogging.test_trade_logger_integration ___________________________________________________________
tests\risk\test_anomaly_detector.py:439: in test_trade_logger_integration
    detector._log_anomaly('TEST', anomaly_result, data, None, AnomalyResponse.LOG_ONLY)
E   AttributeError: 'AnomalyDetector' object has no attribute '_log_anomaly'. Did you mean: 'log_anomalies'?
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
__________________________________________________________________ TestEdgeCases.test_extreme_values __________________________________________________________________
tests\risk\test_anomaly_detector.py:529: in test_extreme_values
    assert result.is_anomaly is True
E   AssertionError: assert True is True
E    +  where True = AnomalyResult(is_anomaly=True, anomaly_type=<AnomalyType.PRICE_GAP: 'price_gap'>, severity=<AnomalySeverity.CRITICAL: ...ose': 100, 'current_close': 1000000, 'gap_pct': 999900.0}, timestamp=datetime.datetime(2025, 9, 23, 10, 4, 59, 444769)).is_anomaly
_____________________________________________________________ TestConfiguration.test_custom_configuration _____________________________________________________________
tests\risk\test_anomaly_detector.py:583: in test_custom_configuration
    detector = AnomalyDetector(config)
risk\anomaly_detector.py:873: in __init__
    self.skip_trade_threshold = self._string_to_severity(self.response_config['skip_trade_threshold'])
E   KeyError: 'skip_trade_threshold'
________________________________________________________ TestConfiguration.test_severity_threshold_conversion _________________________________________________________
tests\risk\test_anomaly_detector.py:593: in test_severity_threshold_conversion
    detector = AnomalyDetector({
risk\anomaly_detector.py:875: in __init__
    self.scale_down_factor = self.response_config['scale_down_factor']
E   KeyError: 'scale_down_factor'
___________________________________________________ TestAnomalyDetection.test_price_zscore_detector_anomalous_data ____________________________________________________
tests\risk\test_risk.py:324: in test_price_zscore_detector_anomalous_data
    assert any(a.anomaly_type == AnomalyType.PRICE_ZSCORE for a in anomalies)
E   assert False
E    +  where False = any(<generator object TestAnomalyDetection.test_price_zscore_detector_anomalous_data.<locals>.<genexpr> at 0x00000258073593F0>)
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
INFO     risk.anomaly_detector:anomaly_detector.py:442 Using conservative anomaly result: cannot calculate z-score
__________________________________________________ TestAnomalyDetection.test_volume_zscore_detector_anomalous_volume __________________________________________________
tests\risk\test_risk.py:356: in test_volume_zscore_detector_anomalous_volume
    assert len(volume_anomalies) > 0
E   assert 0 > 0
E    +  where 0 = len([])
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
INFO     risk.anomaly_detector:anomaly_detector.py:442 Using conservative anomaly result: insufficient data
_______________________________________________________ TestIntegrationTests.test_full_risk_assessment_workflow _______________________________________________________
tests\risk\test_risk.py:459: in test_full_risk_assessment_workflow
    assert result is True
E   assert False is True
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:05:00.010733Z", "level": "TRADE", "module": "crypto_bot", "message": "Signal rejected: error: [<class 'decimal.ConversionSyntax'>]", "correlation_id": null, "request_id": null, "strategy_id": null}
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  risk.risk_manager:risk_manager.py:635 Error checking for anomalies: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
ERROR    risk.risk_manager:risk_manager.py:368 Error evaluating signal: [<class 'decimal.ConversionSyntax'>]
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\risk\risk_manager.py", line 325, in evaluate_signal
    signal.amount = await self.calculate_position_size(signal, market_data)
  File "C:\Users\TU\Desktop\new project\N1V1\risk\risk_manager.py", line 418, in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:05:00.008734+00:00) to int (1758596700008)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestExtremeConditions.test_flash_crash_simulation __________________________________________________________
tests\risk\test_risk.py:540: in test_flash_crash_simulation
    assert AnomalyType.PRICE_ZSCORE in anomaly_types
E   AssertionError: assert <AnomalyType.PRICE_ZSCORE: 'price_zscore'> in {<AnomalyType.PRICE_GAP: 'price_gap'>}
E    +  where <AnomalyType.PRICE_ZSCORE: 'price_zscore'> = AnomalyType.PRICE_ZSCORE
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
___________________________________________________________ TestExtremeConditions.test_extreme_volume_spike ___________________________________________________________
tests\risk\test_risk.py:617: in test_extreme_volume_spike
    assert len(volume_anomalies) > 0
E   assert 0 > 0
E    +  where 0 = len([])
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     risk.anomaly_detector:anomaly_detector.py:890 AnomalyDetector initialized: enabled=True
INFO     risk.anomaly_detector:anomaly_detector.py:442 Using conservative anomaly result: insufficient data
_______________________________________________________ TestRiskManagerEdgeCases.test_risk_manager_zero_balance _______________________________________________________
tests\risk\test_risk.py:698: in test_risk_manager_zero_balance
    result = asyncio.run(manager.calculate_position_size(signal))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:44: in run
    return loop.run_until_complete(main)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________ TestRiskManagerEdgeCases.test_risk_manager_extreme_price_values ___________________________________________________
tests\risk\test_risk.py:719: in test_risk_manager_extreme_price_values
    result = asyncio.run(manager.calculate_position_size(signal))
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\runners.py:44: in run
    return loop.run_until_complete(main)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________ TestRiskManagerAdaptiveIntegration.test_position_sizing_with_adaptive_policy _____________________________________________
tests\risk\test_risk_manager_integration.py:56: in test_position_sizing_with_adaptive_policy
    position_size = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________ TestRiskManagerAdaptiveIntegration.test_adaptive_policy_called_with_correct_data ___________________________________________
tests\risk\test_risk_manager_integration.py:96: in test_adaptive_policy_called_with_correct_data
    position_size = await risk_manager.calculate_position_size(signal, market_data)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestRiskManagerAdaptiveIntegration.test_kill_switch_blocks_trading __________________________________________________
tests\risk\test_risk_manager_integration.py:128: in test_kill_switch_blocks_trading
    position_size = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________ TestRiskManagerAdaptiveIntegration.test_defensive_mode_reduces_position_size _____________________________________________
tests\risk\test_risk_manager_integration.py:158: in test_defensive_mode_reduces_position_size
    position_size = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________ TestRiskManagerAdaptiveIntegration.test_adaptive_atr_position_sizing_with_multiplier _________________________________________
tests\risk\test_risk_manager_integration.py:203: in test_adaptive_atr_position_sizing_with_multiplier
    assert position_size == expected_position
E   AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x0000025806AABDF0> == Decimal('1.0')
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________ TestRiskManagerAdaptiveIntegration.test_error_handling_in_adaptive_integration ____________________________________________
tests\risk\test_risk_manager_integration.py:230: in test_error_handling_in_adaptive_integration
    position_size = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________ TestRiskManagerAdaptiveIntegration.test_different_position_sizing_methods_with_adaptive _______________________________________
tests\risk\test_risk_manager_integration.py:267: in test_different_position_sizing_methods_with_adaptive
    position_size = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________ TestRiskManagerAdaptiveIntegration.test_adaptive_multiplier_application _______________________________________________
tests\risk\test_risk_manager_integration.py:303: in test_adaptive_multiplier_application
    position_size = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________ TestRiskManagerPerformanceIntegration.test_trade_outcome_updates_adaptive_policy ___________________________________________
tests\risk\test_risk_manager_integration.py:343: in test_trade_outcome_updates_adaptive_policy
    position_size = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________ TestRiskManagerPerformanceIntegration.test_consecutive_losses_affect_position_sizing _________________________________________
tests\risk\test_risk_manager_integration.py:377: in test_consecutive_losses_affect_position_sizing
    position_size_1 = await risk_manager.calculate_position_size(signal)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________________ TestAWSKMSKeyManager.test_get_secret_success _____________________________________________________________
venv\lib\site-packages\pytest_asyncio\plugin.py:440: in runtest
    super().runtest()
venv\lib\site-packages\pytest_asyncio\plugin.py:907: in inner
    _loop.run_until_complete(task)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1393: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1431: in __enter__
    self.target = self.getter()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1618: in <lambda>
    getter = lambda: _importer(target)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1257: in _importer
    thing = __import__(import_path)
E   ModuleNotFoundError: No module named 'boto3'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________________ TestAWSKMSKeyManager.test_store_secret_success ____________________________________________________________
venv\lib\site-packages\pytest_asyncio\plugin.py:440: in runtest
    super().runtest()
venv\lib\site-packages\pytest_asyncio\plugin.py:907: in inner
    _loop.run_until_complete(task)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1393: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1431: in __enter__
    self.target = self.getter()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1618: in <lambda>
    getter = lambda: _importer(target)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1257: in _importer
    thing = __import__(import_path)
E   ModuleNotFoundError: No module named 'boto3'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestSecureCredentialManager.test_rotate_key_local __________________________________________________________
tests\security\test_key_management.py:241: in test_rotate_key_local
    assert result is True
E   assert False is True
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________________ TestSecretManager.test_get_secret_vault_success ___________________________________________________________
tests\security\test_secret_manager.py:105: in test_get_secret_vault_success
    result = await get_secret("exchange_api_key")
utils\security.py:732: in get_secret
    raise SecurityException(f"Required secret '{secret_name}' not found in secure storage")
E   utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________________________ TestSecretManager.test_rotate_key_success ______________________________________________________________
tests\security\test_secret_manager.py:141: in test_rotate_key_success
    assert result is True
E   assert False is True
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    security:security.py:251 Security event: key_rotation_failed
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________________ TestAWSKMSKeyManager.test_get_secret_success _____________________________________________________________
tests\security\test_secret_manager.py:389: in test_get_secret_success
    with patch("boto3.Session") as mock_session_class:
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1431: in __enter__
    self.target = self.getter()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1618: in <lambda>
    getter = lambda: _importer(target)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1257: in _importer
    thing = __import__(import_path)
E   ModuleNotFoundError: No module named 'boto3'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________________ TestAWSKMSKeyManager.test_store_secret_success ____________________________________________________________
tests\security\test_secret_manager.py:402: in test_store_secret_success
    with patch("boto3.Session") as mock_session_class:
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1431: in __enter__
    self.target = self.getter()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1618: in <lambda>
    getter = lambda: _importer(target)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1257: in _importer
    thing = __import__(import_path)
E   ModuleNotFoundError: No module named 'boto3'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________________ TestAWSKMSKeyManager.test_health_check_success ____________________________________________________________
tests\security\test_secret_manager.py:415: in test_health_check_success
    with patch("boto3.Session") as mock_session_class:
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1431: in __enter__
    self.target = self.getter()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1618: in <lambda>
    getter = lambda: _importer(target)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1257: in _importer
    thing = __import__(import_path)
E   ModuleNotFoundError: No module named 'boto3'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________ TestBollingerReversionStrategy.test_generate_signals_oversold_reversion _______________________________________________
tests\strategies\test_strategies.py:472: in test_generate_signals_oversold_reversion
    assert len(signals) == 1
E   assert 0 == 1
E    +  where 0 = len([])
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     strategy.bollinger reversion strategy:base_strategy.py:82 Initializing Bollinger Reversion Strategy strategy
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________ TestBollingerReversionStrategy.test_generate_signals_overbought_reversion ______________________________________________
tests\strategies\test_strategies.py:501: in test_generate_signals_overbought_reversion
    assert len(signals) == 1
E   assert 0 == 1
E    +  where 0 = len([])
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     strategy.bollinger reversion strategy:base_strategy.py:82 Initializing Bollinger Reversion Strategy strategy
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________ TestMockingAndIntegration.test_signal_generation_deterministic_timestamps ______________________________________________
tests\strategies\test_strategies.py:1105: in test_signal_generation_deterministic_timestamps
    assert len(signals1) == len(signals2) == 1
E   assert 0 == 1
E    +  where 0 = len([])
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     strategy.rsi strategy:base_strategy.py:82 Initializing RSI Strategy strategy
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________________ TestStrategyGenerator.test_strategy_generation ____________________________________________________________
tests\strategies\test_strategy_generator.py:307: in test_strategy_generation
    assert strategy_class is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    optimization.strategy_generator:strategy_generator.py:403 Strategy class not registered for type: bollinger_reversion
WARNING  optimization.strategy_generator:strategy_generator.py:1539 StrategyFactory failed to create strategy from genome
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________________ TestPerformance.test_generation_performance _____________________________________________________________
tests\strategies\test_strategy_generator.py:624: in test_generation_performance
    assert strategy_class is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    optimization.strategy_generator:strategy_generator.py:389 Genome validation failed: ['Gene 4: Invalid component type StrategyComponent.TIMEFRAME', 'Gene 4: Unknown parameter timeframe']
WARNING  optimization.strategy_generator:strategy_generator.py:1539 StrategyFactory failed to create strategy from genome
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________ TestStrategyIntegrationWorkflow.test_strategy_error_handling_workflow ________________________________________________
tests\strategies\test_strategy_integration.py:269: in test_strategy_error_handling_workflow
    assert failing_fetcher.call_count >= 2
E   assert 0 >= 2
E    +  where 0 = <test_strategy_integration.TestStrategyIntegrationWorkflow.test_strategy_error_handling_workflow.<locals>.FailingDataFetcher object at 0x0000025806E60CA0>.call_count
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     strategy.rsi strategy:base_strategy.py:82 Initializing RSI Strategy strategy
INFO     strategy.rsi strategy:base_strategy.py:132 RSI Strategy strategy initialized
ERROR    strategy.rsi strategy:base_strategy.py:214 Failed to get data for BTC/USDT: Network error
WARNING  strategy.rsi strategy:base_strategy.py:175 No market data available
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________ TestStrategyRiskManagerIntegration.test_risk_manager_position_sizing_integration ___________________________________________
tests\strategies\test_strategy_integration.py:399: in test_risk_manager_position_sizing_integration
    position_size = await risk_manager.calculate_position_size(test_signal, market_dict)
risk\risk_manager.py:418: in calculate_position_size
    adjusted_position = base_position * Decimal(str(risk_multiplier))
E   decimal.InvalidOperation: [<class 'decimal.ConversionSyntax'>]
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     strategy.rsi strategy:base_strategy.py:82 Initializing RSI Strategy strategy
INFO     strategy.rsi strategy:base_strategy.py:132 RSI Strategy strategy initialized
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________ TestStrategyDataFetcherIntegration.test_data_fetcher_error_recovery_integration ___________________________________________
tests\strategies\test_strategy_integration.py:576: in test_data_fetcher_error_recovery_integration
    assert unreliable_fetcher.error_count == unreliable_fetcher.max_errors
E   assert 1 == 2
E    +  where 1 = <test_strategy_integration.TestStrategyDataFetcherIntegration.test_data_fetcher_error_recovery_integration.<locals>.UnreliableDataFetcher object at 0x000002580679C640>.error_count
E    +  and   2 = <test_strategy_integration.TestStrategyDataFetcherIntegration.test_data_fetcher_error_recovery_integration.<locals>.UnreliableDataFetcher object at 0x000002580679C640>.max_errors
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     strategy.rsi strategy:base_strategy.py:82 Initializing RSI Strategy strategy
INFO     strategy.rsi strategy:base_strategy.py:132 RSI Strategy strategy initialized
ERROR    strategy.rsi strategy:base_strategy.py:214 Failed to get data for BTC/USDT: Simulated network error 1
WARNING  strategy.rsi strategy:base_strategy.py:175 No market data available
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestOptimizationIntegration.test_output_validation_backtest_metrics _________________________________________________
tests\test_integration.py:436: in test_output_validation_backtest_metrics
    metrics = compute_backtest_metrics(equity_progression)
backtest\backtester.py:527: in compute_backtest_metrics
    _validate_equity_progression(equity_progression)
backtest\backtester.py:113: in _validate_equity_progression
    raise BacktestValidationError(f"Record {i}: timestamp must be string")
E   backtest.backtester.BacktestValidationError: Record 0: timestamp must be string
________________________________________ TestOptimizationIntegration.test_error_scenario_optimization_with_invalid_parameters _________________________________________
tests\test_integration.py:517: in test_error_scenario_optimization_with_invalid_parameters
    genetic_optimizer.ParameterBounds(
E   AttributeError: 'GeneticOptimizer' object has no attribute 'ParameterBounds'. Did you mean: 'parameter_bounds'?
_______________________________________________________ TestOptimizationIntegration.test_large_dataset_handling _______________________________________________________
tests\test_integration.py:754: in test_large_dataset_handling
    metrics = compute_backtest_metrics(equity_progression)
backtest\backtester.py:527: in compute_backtest_metrics
    _validate_equity_progression(equity_progression)
backtest\backtester.py:113: in _validate_equity_progression
    raise BacktestValidationError(f"Record {i}: timestamp must be string")
E   backtest.backtester.BacktestValidationError: Record 0: timestamp must be string
__________________________________________ TestPerformanceMonitorCorrectness.test_baseline_coefficient_variation_edge_cases ___________________________________________
tests\unit\test_algorithmic_correctness.py:122: in test_baseline_coefficient_variation_edge_cases
    assert isinstance(result, float)
E   assert False
E    +  where False = isinstance(<coroutine object RealTimePerformanceMonitor._calculate_system_health_score at 0x0000025806C0FB50>, float)
__________________________________________________ TestPerformanceMonitorCorrectness.test_anomaly_detection_zero_std __________________________________________________
tests\unit\test_algorithmic_correctness.py:143: in test_anomaly_detection_zero_std
    assert len(anomalies) == 0
E   TypeError: object of type 'coroutine' has no len()
_____________________________________________ TestPerformanceMonitorCorrectness.test_percentile_anomaly_score_calculation _____________________________________________
tests\unit\test_algorithmic_correctness.py:170: in test_percentile_anomaly_score_calculation
    assert len(anomalies) > 0
E   TypeError: object of type 'coroutine' has no len()
__________________________________________________ TestOrderManagerPrecision.test_decimal_initial_balance_conversion __________________________________________________
tests\unit\test_algorithmic_correctness.py:267: in test_decimal_initial_balance_conversion
    assert isinstance(order_manager.paper_executor.initial_balance, Decimal)
E   AssertionError: assert False
E    +  where False = isinstance(<MagicMock name='PaperOrderExecutor().initial_balance' id='2577060123248'>, Decimal)
E    +    where <MagicMock name='PaperOrderExecutor().initial_balance' id='2577060123248'> = <MagicMock name='PaperOrderExecutor()' id='2577060123632'>.initial_balance
E    +      where <MagicMock name='PaperOrderExecutor()' id='2577060123632'> = <core.order_manager.OrderManager object at 0x0000025804C0C2B0>.paper_executor
___________________________________________________ TestOrderManagerPrecision.test_invalid_initial_balance_fallback ___________________________________________________
tests\unit\test_algorithmic_correctness.py:289: in test_invalid_initial_balance_fallback
    assert isinstance(order_manager.paper_executor.initial_balance, Decimal)
E   AssertionError: assert False
E    +  where False = isinstance(<MagicMock name='PaperOrderExecutor().initial_balance' id='2577092847136'>, Decimal)
E    +    where <MagicMock name='PaperOrderExecutor().initial_balance' id='2577092847136'> = <MagicMock name='PaperOrderExecutor()' id='2577092847376'>.initial_balance
E    +      where <MagicMock name='PaperOrderExecutor()' id='2577092847376'> = <core.order_manager.OrderManager object at 0x0000025806B40160>.paper_executor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Invalid initial_balance value: invalid, using default
____________________________________________________ TestOrderManagerPrecision.test_none_initial_balance_fallback _____________________________________________________
tests\unit\test_algorithmic_correctness.py:311: in test_none_initial_balance_fallback
    assert isinstance(order_manager.paper_executor.initial_balance, Decimal)
E   AssertionError: assert False
E    +  where False = isinstance(<MagicMock name='PaperOrderExecutor().initial_balance' id='2577092833632'>, Decimal)
E    +    where <MagicMock name='PaperOrderExecutor().initial_balance' id='2577092833632'> = <MagicMock name='PaperOrderExecutor()' id='2577092824944'>.initial_balance
E    +      where <MagicMock name='PaperOrderExecutor()' id='2577092824944'> = <core.order_manager.OrderManager object at 0x0000025806B3E620>.paper_executor
___________________________________________________ TestStatisticalEdgeCases.test_percentile_calculation_robustness ___________________________________________________
tests\unit\test_algorithmic_correctness.py:352: in test_percentile_calculation_robustness
    assert p95 >= p99  # 95th percentile should be <= 99th percentile
E   assert 4.8 >= 4.96
____________________________________________________ TestCustomExceptionMiddleware.test_normal_request_passthrough ____________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'mock' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_api_app.py:78: in test_normal_request_passthrough
    send.assert_called_once()
E   AssertionError: Expected 'mock' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________________ TestAPIEndpoints.test_health_endpoint_no_bot_engine _________________________________________________________
tests\unit\test_api_app.py:338: in test_health_endpoint_no_bot_engine
    assert response.status_code == 200
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
________________________________________ TestBinaryModelIntegrationEdgeCases.test_binary_model_prediction_with_exact_threshold ________________________________________
tests\unit\test_binary_integration.py:560: in test_binary_model_prediction_with_exact_threshold
    assert result.should_trade == True  # Should trade at exact threshold
E   assert False == True
E    +  where False = BinaryModelResult(should_trade=False, probability=-1.0, confidence=0.0, threshold=0.6, features={}, timestamp=datetime.datetime(2025, 9, 23, 10, 5, 20, 517566)).should_trade
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________ TestBinaryModelIntegrationEdgeCases.test_binary_model_prediction_with_low_confidence _________________________________________
tests\unit\test_binary_integration.py:591: in test_binary_model_prediction_with_low_confidence
    assert result.probability == 0.55
E   assert -1.0 == 0.55
E    +  where -1.0 = BinaryModelResult(should_trade=False, probability=-1.0, confidence=0.0, threshold=0.6, features={}, timestamp=datetime.datetime(2025, 9, 23, 10, 5, 20, 603525)).probability
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________ TestGlobalIntegrationFunctions.test_integrate_binary_model_convenience_function ___________________________________________
tests\unit\test_binary_integration.py:994: in test_integrate_binary_model_convenience_function
    result = await integrate_binary_model(market_data, "BTC/USDT")
E   NameError: name 'integrate_binary_model' is not defined
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________ TestErrorRecovery.test_binary_model_prediction_error_recovery ____________________________________________________
tests\unit\test_binary_integration.py:1069: in test_binary_model_prediction_error_recovery
    assert result.probability == 0.0
E   assert -1.0 == 0.0
E    +  where -1.0 = BinaryModelResult(should_trade=False, probability=-1.0, confidence=0.0, threshold=0.6, features={}, timestamp=datetime.datetime(2025, 9, 23, 10, 5, 21, 167125)).probability
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________________ TestErrorRecovery.test_strategy_selection_error_recovery _______________________________________________________
tests\unit\test_binary_integration.py:1101: in test_strategy_selection_error_recovery
    assert result.regime == "UNKNOWN"
E   AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == 'UNKNOWN'
E    +  where <MarketRegime.UNKNOWN: 'unknown'> = StrategySelectionResult(selected_strategy=None, direction='neutral', regime=<MarketRegime.UNKNOWN: 'unknown'>, confidence=0.0, reasoning='Strategy selection failed', risk_multiplier=1.0).regime
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Strategy selection failed: Regime detection failed
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________________ TestErrorRecovery.test_complete_pipeline_error_recovery _______________________________________________________
tests\unit\test_binary_integration.py:1133: in test_complete_pipeline_error_recovery
    assert "Integration error" in decision.reasoning
E   AssertionError: assert 'Integration error' in 'Binary integration disabled'
E    +  where 'Binary integration disabled' = IntegratedTradingDecision(should_trade=False, binary_probability=0.0, selected_strategy=None, direction='neutral', reg..., risk_score=1.0, reasoning='Binary integration disabled', timestamp=datetime.datetime(2025, 9, 23, 10, 5, 21, 359381)).reasoning
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________ TestPerformanceAndScalability.test_concurrent_market_data_processing _________________________________________________
tests\unit\test_binary_integration.py:1193: in test_concurrent_market_data_processing
    assert result.should_trade == True
E   AssertionError: assert False == True
E    +  where False = IntegratedTradingDecision(should_trade=False, binary_probability=0.0, selected_strategy=None, direction='neutral', reg..., risk_score=1.0, reasoning='Binary integration disabled', timestamp=datetime.datetime(2025, 9, 23, 10, 5, 21, 443534)).should_trade
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________________ TestLoggingAndMonitoring.test_binary_prediction_logging _______________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_binary_integration.py:1239: in test_binary_prediction_logging
    mock_trade_logger.log_binary_prediction.assert_called_once()
E   AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________ TestBinaryModelIntegrationFeatureExtraction.test_calculate_macd_edge_cases ______________________________________________
tests\unit\test_binary_integration_enhanced.py:180: in test_calculate_macd_edge_cases
    assert macd == 0.0  # Should return fallback
E   assert 0.02243589743588359 == 0.0
__________________________________________ TestBinaryModelIntegrationGlobalFunctions.test_get_binary_integration_with_config __________________________________________
tests\unit\test_binary_integration_enhanced.py:241: in test_get_binary_integration_with_config
    assert instance.enabled == True
E   assert False == True
E    +  where False = <core.binary_model_integration.BinaryModelIntegration object at 0x0000025806E973A0>.enabled
__________________________________________ TestBinaryModelIntegrationMetricsIntegration.test_metrics_recording_in_prediction __________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'record_prediction' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_binary_integration_enhanced.py:289: in test_metrics_recording_in_prediction
    mock_collector.record_prediction.assert_called_once()
E   AssertionError: Expected 'record_prediction' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________ TestBinaryModelIntegrationErrorRecovery.test_predict_binary_model_with_model_exception ________________________________________
tests\unit\test_binary_integration_enhanced.py:359: in test_predict_binary_model_with_model_exception
    assert result.probability == 0.0
E   assert -1.0 == 0.0
E    +  where -1.0 = BinaryModelResult(should_trade=False, probability=-1.0, confidence=0.0, threshold=0.6, features={}, timestamp=datetime.datetime(2025, 9, 23, 10, 5, 55, 133924)).probability
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________ TestBinaryModelIntegrationErrorRecovery.test_select_strategy_with_regime_detector_failure ______________________________________
tests\unit\test_binary_integration_enhanced.py:388: in test_select_strategy_with_regime_detector_failure
    assert result.regime == MarketRegime.UNKNOWN
E   AssertionError: assert <MarketRegime.UNKNOWN: 'unknown'> == <MarketRegime.UNKNOWN: 'unknown'>
E    +  where <MarketRegime.UNKNOWN: 'unknown'> = StrategySelectionResult(selected_strategy=None, direction='neutral', regime=<MarketRegime.UNKNOWN: 'unknown'>, confidence=0.0, reasoning='Strategy selection failed', risk_multiplier=1.0).regime
E    +  and   <MarketRegime.UNKNOWN: 'unknown'> = MarketRegime.UNKNOWN
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Strategy selection failed: Regime detection failed
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________ TestBinaryModelIntegrationLogging.test_binary_prediction_logging_integration _____________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_binary_integration_enhanced.py:554: in test_binary_prediction_logging_integration
    mock_trade_logger.log_binary_prediction.assert_called_once()
E   AssertionError: Expected 'log_binary_prediction' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________ TestMetricsCollection.test_collect_binary_model_metrics_with_exception ________________________________________________
tests\unit\test_binary_model_metrics.py:512: in test_collect_binary_model_metrics_with_exception
    assert mock_metrics_collector.record_metric.called
E   AssertionError: assert False
E    +  where False = <AsyncMock name='mock.record_metric' id='2577090139744'>.called
E    +    where <AsyncMock name='mock.record_metric' id='2577090139744'> = <Mock spec='MetricsCollector' id='2577090149344'>.record_metric
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________ TestAlertGeneration.test_check_for_alerts_trade_frequency_drift ___________________________________________________
tests\unit\test_binary_model_metrics.py:574: in test_check_for_alerts_trade_frequency_drift
    assert "trade frequency changed" in alert["description"]
E   AssertionError: assert 'trade frequency changed' in 'Trade frequency changed by 100.0%'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Error checking for binary model alerts: 'symbol'
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________________ TestAlertGeneration.test_check_for_alerts_accuracy_drop _______________________________________________________
tests\unit\test_binary_model_metrics.py:584: in test_check_for_alerts_accuracy_drop
    assert len(accuracy_alerts) == 1
E   assert 0 == 1
E    +  where 0 = len([])
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Error checking for binary model alerts: 'symbol'
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestPerformanceReporting.test_get_performance_report _________________________________________________________
tests\unit\test_binary_model_metrics.py:681: in test_get_performance_report
    report = collector.get_performance_report()
core\binary_model_metrics.py:563: in get_performance_report
    "hit_rate_metrics": self._calculate_hit_rate_metrics(),
core\binary_model_metrics.py:293: in _calculate_hit_rate_metrics
    correct_decisions = sum(1 for d in recent_decisions if d["was_correct"])
core\binary_model_metrics.py:293: in <genexpr>
    correct_decisions = sum(1 for d in recent_decisions if d["was_correct"])
E   KeyError: 'was_correct'
_____________________________________________ TestErrorHandling.test_collect_metrics_with_binary_integration_import_error _____________________________________________
tests\unit\test_binary_model_metrics.py:746: in test_collect_metrics_with_binary_integration_import_error
    with patch('core.binary_model_metrics.get_binary_integration', side_effect=ImportError):
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'core.binary_model_metrics' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\binary_model_metrics.py'> does not have the attribute 'get_binary_integration'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________ TestErrorHandling.test_record_prediction_with_invalid_data ______________________________________________________
tests\unit\test_binary_model_metrics.py:755: in test_record_prediction_with_invalid_data
    collector.record_prediction(
core\binary_model_metrics.py:173: in record_prediction
    "decision": "trade" if probability >= threshold else "skip"
E   TypeError: '>=' not supported between instances of 'str' and 'float'
__________________________________________________ TestErrorHandling.test_record_decision_outcome_with_invalid_data ___________________________________________________
tests\unit\test_binary_model_metrics.py:771: in test_record_decision_outcome_with_invalid_data
    collector.record_decision_outcome(
core\binary_model_metrics.py:232: in record_decision_outcome
    regime_stats["total_pnl"] += pnl
E   TypeError: unsupported operand type(s) for +=: 'float' and 'str'
____________________________________________________ TestErrorHandling.test_calculate_metrics_with_corrupted_data _____________________________________________________
tests\unit\test_binary_model_metrics.py:797: in test_calculate_metrics_with_corrupted_data
    avg_ptrade = collector._calculate_average_ptrade()
core\binary_model_metrics.py:247: in _calculate_average_ptrade
    recent_predictions = [
core\binary_model_metrics.py:249: in <listcomp>
    if p["timestamp"] > cutoff_time
E   TypeError: '>' not supported between instances of 'str' and 'datetime.datetime'
______________________________________________________ TestDataIntegrity.test_prediction_history_data_integrity _______________________________________________________
tests\unit\test_binary_model_metrics.py:841: in test_prediction_history_data_integrity
    collector.record_prediction(**prediction)
E   TypeError: BinaryModelMetricsCollector.record_prediction() got an unexpected keyword argument 'decision'
__________________________________________________________________ TestBotEngine.test_trading_cycle ___________________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'get_historical_data' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_bot_engine.py:91: in test_trading_cycle
    mock_data_fetcher.get_historical_data.assert_called_once()
E   AssertionError: Expected 'get_historical_data' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestBotEngineInitialization.test_initialization_with_minimal_config _________________________________________________
tests\unit\test_bot_engine_comprehensive.py:87: in test_initialization_with_minimal_config
    assert engine.pairs == ["BTC/USDT"]
E   AssertionError: assert [] == ['BTC/USDT']
E
E     Right contains one more item: 'BTC/USDT'
E     Use -v to get more diff
___________________________________________________ TestBotEngineInitialization.test_initialization_portfolio_mode ____________________________________________________
tests\unit\test_bot_engine_comprehensive.py:101: in test_initialization_portfolio_mode
    assert set(engine.pairs) == {"BTC/USDT", "ETH/USDT"}
E   AssertionError: assert set() == {'BTC/USDT', 'ETH/USDT'}
E
E     Extra items in the right set:
E     'BTC/USDT'
E     'ETH/USDT'
E     Use -v to get more diff
________________________________________________ TestBotEngineInitialization.test_initialization_with_invalid_balance _________________________________________________
tests\unit\test_bot_engine_comprehensive.py:110: in test_initialization_with_invalid_balance
    assert engine.starting_balance == 1000.0
E   assert 10000.0 == 1000.0
E    +  where 10000.0 = <core.bot_engine.BotEngine object at 0x0000025806635510>.starting_balance
_________________________________________________ TestBotEngineInitialization.test_initialization_with_empty_markets __________________________________________________
tests\unit\test_bot_engine_comprehensive.py:119: in test_initialization_with_empty_markets
    assert engine.pairs == ["BTC/USDT"]
E   AssertionError: assert [] == ['BTC/USDT']
E
E     Right contains one more item: 'BTC/USDT'
E     Use -v to get more diff
____________________________________________________ TestBotEnginePerformanceTracking.test_calculate_max_drawdown _____________________________________________________
tests\unit\test_bot_engine_comprehensive.py:518: in test_calculate_max_drawdown
    assert mock_engine.performance_stats["max_drawdown"] == 0.08
E   assert 0.061224489795918366 == 0.08
__________________________________________________ TestBotEnginePerformanceTracking.test_record_trade_equity_success __________________________________________________
tests\unit\test_bot_engine_comprehensive.py:534: in test_record_trade_equity_success
    mock_engine.order_manager.get_equity = AsyncMock(return_value=10100.0)
E   AttributeError: 'NoneType' object has no attribute 'get_equity'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________ TestBotEnginePerformanceTracking.test_record_trade_equity_missing_equity _______________________________________________
tests\unit\test_bot_engine_comprehensive.py:556: in test_record_trade_equity_missing_equity
    mock_engine.order_manager.get_equity = AsyncMock(return_value=0.0)
E   AttributeError: 'NoneType' object has no attribute 'get_equity'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________ TestBotEngineStateManagement.test_run_main_loop_normal_operation ___________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected '_update_display' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_bot_engine_comprehensive.py:655: in test_run_main_loop_normal_operation
    mock_update.assert_called_once()
E   AssertionError: Expected '_update_display' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________ TestBotEngineIntegration.test_process_binary_integration_enabled ___________________________________________________
tests\unit\test_bot_engine_comprehensive.py:807: in test_process_binary_integration_enabled
    assert len(decisions) == 1
E   assert 0 == 1
E    +  where 0 = len([])
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________ TestBotEngineIntegration.test_execute_integrated_decisions ______________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'execute_order' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_bot_engine_comprehensive.py:848: in test_execute_integrated_decisions
    mock_engine.order_manager.execute_order.assert_called_once()
E   AssertionError: Expected 'execute_order' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Error creating signal from decision: TradingSignal.__init__() got an unexpected keyword argument 'strength'
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestBotEngineErrorHandling.test_trading_cycle_with_data_fetch_error _________________________________________________
tests\unit\test_bot_engine_comprehensive.py:904: in test_trading_cycle_with_data_fetch_error
    await mock_engine._trading_cycle()
core\bot_engine.py:375: in _trading_cycle
    await self._update_state()
core\bot_engine.py:683: in _update_state
    self.state.balance = await self.order_manager.get_balance()
E   AttributeError: 'NoneType' object has no attribute 'get_balance'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________ TestBotEngineErrorHandling.test_signal_generation_with_strategy_error ________________________________________________
tests\unit\test_bot_engine_comprehensive.py:921: in test_signal_generation_with_strategy_error
    signals = await mock_engine._generate_signals(market_data)
core\bot_engine.py:621: in _generate_signals
    signals = await self._generate_signals_from_all_strategies(market_data)
core\bot_engine.py:632: in _generate_signals_from_all_strategies
    strategy_signals = await strategy.generate_signals(market_data, multi_tf_data)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2234: in _execute_mock_call
    raise effect
E   Exception: Strategy failed
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________ TestBotEngineErrorHandling.test_risk_evaluation_with_component_error _________________________________________________
tests\unit\test_bot_engine_comprehensive.py:929: in test_risk_evaluation_with_component_error
    mock_engine.risk_manager.evaluate_signal = AsyncMock(side_effect=Exception("Risk evaluation failed"))
E   AttributeError: 'NoneType' object has no attribute 'evaluate_signal'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________ TestBotEngineErrorHandling.test_order_execution_with_component_error _________________________________________________
tests\unit\test_bot_engine_comprehensive.py:943: in test_order_execution_with_component_error
    mock_engine.order_manager.execute_order = AsyncMock(side_effect=Exception("Order execution failed"))
E   AttributeError: 'NoneType' object has no attribute 'execute_order'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestBotEngineErrorHandling.test_state_update_with_component_errors __________________________________________________
tests\unit\test_bot_engine_comprehensive.py:967: in test_state_update_with_component_errors
    mock_engine.order_manager.get_balance = AsyncMock(side_effect=Exception("Balance fetch failed"))
E   AttributeError: 'NoneType' object has no attribute 'get_balance'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________ TestBotEngineEdgeCases.test_trading_cycle_with_empty_market_data ___________________________________________________
tests\unit\test_bot_engine_comprehensive.py:1019: in test_trading_cycle_with_empty_market_data
    await mock_engine._trading_cycle()
core\bot_engine.py:375: in _trading_cycle
    await self._update_state()
core\bot_engine.py:683: in _update_state
    self.state.balance = await self.order_manager.get_balance()
E   AttributeError: 'NoneType' object has no attribute 'get_balance'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________ TestBotEngineEdgeCases.test_trading_cycle_with_none_market_data ___________________________________________________
tests\unit\test_bot_engine_comprehensive.py:1030: in test_trading_cycle_with_none_market_data
    await mock_engine._trading_cycle()
core\bot_engine.py:375: in _trading_cycle
    await self._update_state()
core\bot_engine.py:683: in _update_state
    self.state.balance = await self.order_manager.get_balance()
E   AttributeError: 'NoneType' object has no attribute 'get_balance'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestBotEngineEdgeCases.test_concurrent_trading_cycles ________________________________________________________
tests\unit\test_bot_engine_comprehensive.py:1060: in test_concurrent_trading_cycles
    await asyncio.gather(*tasks)
core\bot_engine.py:375: in _trading_cycle
    await self._update_state()
core\bot_engine.py:683: in _update_state
    self.state.balance = await self.order_manager.get_balance()
E   AttributeError: 'NoneType' object has no attribute 'get_balance'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________________ TestCacheBatchOperations.test_set_multiple_ohlcv_success _______________________________________________________
tests\unit\test_cache_comprehensive.py:386: in test_set_multiple_ohlcv_success
    assert result["BTC/USDT"] == True
E   assert False == True
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    cache:logging_utils.py:181 Batch OHLCV cache set failed: 'coroutine' object has no attribute 'setex'
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________ TestCacheInvalidation.test_invalidate_symbol_data_specific_types ___________________________________________________
tests\unit\test_cache_comprehensive.py:461: in test_invalidate_symbol_data_specific_types
    assert result == 1  # One key deleted
E   assert 3 == 1
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     cache:logging_utils.py:167 Invalidated 3 cache entries for symbol BTC/USDT
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestCacheContextManager.test_context_manager_success _________________________________________________________
tests\unit\test_cache_comprehensive.py:607: in test_context_manager_success
    async with CacheContext(cache_config) as cache:
core\cache.py:848: in __aenter__
    raise RuntimeError("Cache not initialized and no config provided")
E   RuntimeError: Cache not initialized and no config provided
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestCacheErrorHandling.test_batch_operations_with_partial_failures __________________________________________________
tests\unit\test_cache_comprehensive.py:690: in test_batch_operations_with_partial_failures
    assert "ETH/USDT" in result
E   AssertionError: assert 'ETH/USDT' in {'ADA/USDT': None, 'BTC/USDT': {'price': 50000}}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    cache:logging_utils.py:160 Failed to deserialize cached OHLCV for ETH/USDT
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________________ TestCacheEviction.test_evict_expired_entries _____________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:898: in assert_called
    raise AssertionError(msg)
E   AssertionError: Expected 'delete' to have been called.

During handling of the above exception, another exception occurred:
tests\unit\test_cache_eviction.py:110: in test_evict_expired_entries
    mock_redis.delete.assert_called()
E   AssertionError: Expected 'delete' to have been called.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    cache:logging_utils.py:181 Failed to evict expired entries: 'int' object is not iterable
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestCacheIntegration.test_memory_manager_integration _________________________________________________________
tests\unit\test_cache_eviction.py:366: in test_memory_manager_integration
    memory_manager = MemoryManager(enable_monitoring=False)
core\memory_manager.py:36: in __init__
    config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
_______________________________________________ TestConfigManagerValidation.test_config_manager_init_with_valid_config ________________________________________________
tests\unit\test_config_manager.py:377: in test_config_manager_init_with_valid_config
    manager = ConfigManager(self.config_file)
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:239: in _load_config
    self._load_from_environment()
core\config_manager.py:321: in _load_from_environment
    loop.run_until_complete(self._load_secrets_securely())
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
core\config_manager.py:330: in _load_secrets_securely
    api_key = await get_secret("exchange_api_key")
utils\security.py:732: in get_secret
    raise SecurityException(f"Required secret '{secret_name}' not found in secure storage")
E   utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Failed to load required secrets in production mode: Required secret 'exchange_api_key' not found in secure storage
_____________________________________________ TestConfigManagerValidation.test_config_manager_init_with_missing_sections ______________________________________________
tests\unit\test_config_manager.py:428: in test_config_manager_init_with_missing_sections
    manager = ConfigManager(self.config_file)
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:239: in _load_config
    self._load_from_environment()
core\config_manager.py:321: in _load_from_environment
    loop.run_until_complete(self._load_secrets_securely())
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
core\config_manager.py:330: in _load_secrets_securely
    api_key = await get_secret("exchange_api_key")
utils\security.py:732: in get_secret
    raise SecurityException(f"Required secret '{secret_name}' not found in secure storage")
E   utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Failed to load required secrets in production mode: Required secret 'exchange_api_key' not found in secure storage
_________________________________________________ TestDataProcessorSecurity.test_calculate_rsi_batch_data_validation __________________________________________________
tests\unit\test_core_security.py:60: in test_calculate_rsi_batch_data_validation
    assert "TEST" not in result  # Missing column should be skipped
E   AssertionError: assert 'TEST' not in {'TEST':    open\n0   100\n1   101\n2   102}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Invalid symbol: 123, skipping
Invalid data type for TEST: expected DataFrame, got <class 'str'>
Insufficient data for TEST: 3 < 14
__________________________________________________________ TestMetricsEndpointSecurity.test_secure_defaults ___________________________________________________________
tests\unit\test_core_security.py:122: in test_secure_defaults
    assert endpoint.enable_auth == True  # Secure default
E   assert False == True
E    +  where False = <core.metrics_endpoint.MetricsEndpoint object at 0x00000258085BE3E0>.enable_auth
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     metrics_endpoint:logging_utils.py:167 MetricsEndpoint initialized on 0.0.0.0:9090/metrics | component=metrics_endpoint | operation=initialize | host=0.0.0.0 | port=9090 | path=/metrics
_____________________________________ TestCircuitBreakerMonitoringIntegration.test_monitoring_performance_during_circuit_breaker ______________________________________
tests\unit\test_cross_feature_integration.py:194: in test_monitoring_performance_during_circuit_breaker
    assert degradation < 0.5, f"Performance degraded by {degradation:.1%} during circuit breaker (threshold: 50%)"
E   AssertionError: Performance degraded by 54.4% during circuit breaker (threshold: 50%)
E   assert 0.5437491496951923 < 0.5
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________________ TestDependencyInjection.test_component_factory_creation _______________________________________________________
tests\unit\test_dependency_injection.py:63: in test_component_factory_creation
    factory = ComponentFactory()
core\component_factory.py:42: in __init__
    self._config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
________________________________________________________ TestDependencyInjection.test_cache_component_creation ________________________________________________________
tests\unit\test_dependency_injection.py:79: in test_cache_component_creation
    factory = ComponentFactory()
core\component_factory.py:42: in __init__
    self._config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
________________________________________________________ TestDependencyInjection.test_memory_manager_creation _________________________________________________________
tests\unit\test_dependency_injection.py:89: in test_memory_manager_creation
    factory = ComponentFactory()
core\component_factory.py:42: in __init__
    self._config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
___________________________________________________________ TestDependencyInjection.test_component_caching ____________________________________________________________
tests\unit\test_dependency_injection.py:98: in test_component_caching
    factory = ComponentFactory()
core\component_factory.py:42: in __init__
    self._config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
_________________________________________________________ TestDependencyInjection.test_configuration_override _________________________________________________________
tests\unit\test_dependency_injection.py:113: in test_configuration_override
    config_manager.set("cache.ttl_config.market_ticker", 5)
core\config_manager.py:434: in set
    self._apply_overrides()
core\config_manager.py:386: in _apply_overrides
    self._set_nested_value(self._config, key.split("."), value)
core\config_manager.py:398: in _set_nested_value
    self._set_nested_value(attr, keys[1:], value)
core\config_manager.py:398: in _set_nested_value
    self._set_nested_value(attr, keys[1:], value)
core\config_manager.py:391: in _set_nested_value
    setattr(obj, keys[0], value)
E   AttributeError: 'dict' object has no attribute 'market_ticker'
_______________________________________________________ TestDependencyInjection.test_async_component_operations _______________________________________________________
tests\unit\test_dependency_injection.py:130: in test_async_component_operations
    factory = ComponentFactory()
core\component_factory.py:42: in __init__
    self._config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestDependencyInjection.test_factory_global_instance _________________________________________________________
tests\unit\test_dependency_injection.py:144: in test_factory_global_instance
    factory1 = get_component_factory()
core\component_factory.py:311: in get_component_factory
    _component_factory = ComponentFactory()
core\component_factory.py:42: in __init__
    self._config_manager = get_config_manager()
core\config_manager.py:683: in get_config_manager
    _config_manager = ConfigManager()
core\config_manager.py:182: in __init__
    self._load_config()
core\config_manager.py:226: in _load_config
    raise ValueError(error_msg)
E   ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Configuration validation error: Field 'trading': Extra inputs are not permitted
Configuration validation error: Field 'order': Extra inputs are not permitted
Configuration validation error: Field 'risk': Extra inputs are not permitted
Configuration validation error: Field 'backtesting': Extra inputs are not permitted
Configuration validation error: Field 'notifications': Extra inputs are not permitted
Configuration validation error: Field 'logging': Extra inputs are not permitted
Configuration validation error: Field 'advanced': Extra inputs are not permitted
Configuration validation error: Field 'profiling': Extra inputs are not permitted
Configuration validation error: Field 'queue': Extra inputs are not permitted
Configuration validation error: Field 'event_bus': Extra inputs are not permitted
Configuration validation error: Field 'diagnostics': Extra inputs are not permitted
Configuration validation error: Field 'knowledge_base': Extra inputs are not permitted
Configuration validation error: Field 'strategy_selector': Extra inputs are not permitted
Configuration validation error: Field 'multi_timeframe': Extra inputs are not permitted
Configuration validation error: Field 'regime_forecasting': Extra inputs are not permitted
Configuration validation error: Field 'strategy_generator': Extra inputs are not permitted
Configuration validation error: Field 'ensemble': Extra inputs are not permitted
Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra inputs are not permitted; Field 'backtesting': Extra inputs are not permitted; Field 'notifications': Extra inputs are not permitted; Field 'logging': Extra inputs are not permitted; Field 'advanced': Extra inputs are not permitted; Field 'profiling': Extra inputs are not permitted; Field 'queue': Extra inputs are not permitted; Field 'event_bus': Extra inputs are not permitted; Field 'diagnostics': Extra inputs are not permitted; Field 'knowledge_base': Extra inputs are not permitted; Field 'strategy_selector': Extra inputs are not permitted; Field 'multi_timeframe': Extra inputs are not permitted; Field 'regime_forecasting': Extra inputs are not permitted; Field 'strategy_generator': Extra inputs are not permitted; Field 'ensemble': Extra inputs are not permitted
______________________________________________ TestConfigurationIntegration.test_performance_tracker_config_integration _______________________________________________
tests\unit\test_dependency_injection.py:186: in test_performance_tracker_config_integration
    assert perf_tracker.starting_balance == 2000.0
E   assert 1000.0 == 2000.0
E    +  where 1000.0 = <core.performance_tracker.PerformanceTracker object at 0x0000025806AF0940>.starting_balance
_________________________________________________________ TestConfigurationPersistence.test_config_save_load __________________________________________________________
tests\unit\test_dependency_injection.py:247: in test_config_save_load
    config_manager.set("cache.ttl_config.market_ticker", 10)
core\config_manager.py:434: in set
    self._apply_overrides()
core\config_manager.py:386: in _apply_overrides
    self._set_nested_value(self._config, key.split("."), value)
core\config_manager.py:398: in _set_nested_value
    self._set_nested_value(attr, keys[1:], value)
core\config_manager.py:398: in _set_nested_value
    self._set_nested_value(attr, keys[1:], value)
core\config_manager.py:391: in _set_nested_value
    setattr(obj, keys[0], value)
E   AttributeError: 'dict' object has no attribute 'market_ticker'
_________________________________________________________ TestHealthEndpoint.test_health_endpoint_returns_200 _________________________________________________________
tests\unit\test_endpoints.py:82: in test_health_endpoint_returns_200
    assert response.status_code == 200
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
________________________________________________________ TestHealthEndpoint.test_health_endpoint_returns_json _________________________________________________________
tests\unit\test_endpoints.py:88: in test_health_endpoint_returns_json
    assert "status" in data
E   AssertionError: assert 'status' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/health'}, 'message': 'An unexpected error occurred'}}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
______________________________________________________ TestHealthEndpoint.test_health_endpoint_returns_metadata _______________________________________________________
tests\unit\test_endpoints.py:103: in test_health_endpoint_returns_metadata
    assert isinstance(data["version"], str)
E   KeyError: 'version'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
____________________________________________________ TestHealthEndpoint.test_health_endpoint_correlation_id_unique ____________________________________________________
tests\unit\test_endpoints.py:118: in test_health_endpoint_correlation_id_unique
    assert data1["correlation_id"] != data2["correlation_id"]
E   KeyError: 'correlation_id'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
______________________________________________________ TestHealthEndpoint.test_health_endpoint_timestamp_format _______________________________________________________
tests\unit\test_endpoints.py:132: in test_health_endpoint_timestamp_format
    assert re.match(iso_pattern, data["timestamp"])
E   KeyError: 'timestamp'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
_______________________________________________ TestReadinessEndpoint.test_readiness_endpoint_returns_200_when_healthy ________________________________________________
tests\unit\test_endpoints.py:146: in test_readiness_endpoint_returns_200_when_healthy
    assert response.status_code in [200, 503]  # 503 if dependencies fail
E   assert 500 in [200, 503]
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_returns_json_structure _________________________________________________
tests\unit\test_endpoints.py:153: in test_readiness_endpoint_returns_json_structure
    assert "ready" in data
E   AssertionError: assert 'ready' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
_____________________________________________ TestReadinessEndpoint.test_readiness_endpoint_includes_all_check_components _____________________________________________
tests\unit\test_endpoints.py:169: in test_readiness_endpoint_includes_all_check_components
    checks = data["checks"]
E   KeyError: 'checks'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
_________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_correlation_id_unique _________________________________________________
tests\unit\test_endpoints.py:186: in test_readiness_endpoint_correlation_id_unique
    assert data1["correlation_id"] != data2["correlation_id"]
E   KeyError: 'correlation_id'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
___________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_timestamp_format ____________________________________________________
tests\unit\test_endpoints.py:200: in test_readiness_endpoint_timestamp_format
    assert re.match(iso_pattern, data["timestamp"])
E   KeyError: 'timestamp'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
________________________________________ TestReadinessEndpoint.test_readiness_endpoint_returns_503_when_bot_engine_unavailable ________________________________________
tests\unit\test_endpoints.py:215: in test_readiness_endpoint_returns_503_when_bot_engine_unavailable
    assert response.status_code == 503
E   assert 500 == 503
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_check_details_structure ________________________________________________
tests\unit\test_endpoints.py:230: in test_readiness_endpoint_check_details_structure
    for component, check_data in data["checks"].items():
E   KeyError: 'checks'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
__________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_latency_measurement __________________________________________________
tests\unit\test_endpoints.py:247: in test_readiness_endpoint_latency_measurement
    assert data["total_latency_ms"] >= 0
E   KeyError: 'total_latency_ms'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
________________________________________ TestReadinessEndpoint.test_readiness_endpoint_handles_missing_env_vars[DATABASE_URL] _________________________________________
tests\unit\test_endpoints.py:276: in test_readiness_endpoint_handles_missing_env_vars
    assert "checks" in data
E   AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
______________________________________ TestReadinessEndpoint.test_readiness_endpoint_handles_missing_env_vars[EXCHANGE_API_URL] _______________________________________
tests\unit\test_endpoints.py:276: in test_readiness_endpoint_handles_missing_env_vars
    assert "checks" in data
E   AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
______________________________________ TestReadinessEndpoint.test_readiness_endpoint_handles_missing_env_vars[MESSAGE_QUEUE_URL] ______________________________________
tests\unit\test_endpoints.py:276: in test_readiness_endpoint_handles_missing_env_vars
    assert "checks" in data
E   AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
__________________________________________ TestReadinessEndpoint.test_readiness_endpoint_handles_missing_env_vars[REDIS_URL] __________________________________________
tests\unit\test_endpoints.py:276: in test_readiness_endpoint_handles_missing_env_vars
    assert "checks" in data
E   AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
___________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_bot_engine_check ____________________________________________________
tests\unit\test_endpoints.py:296: in test_readiness_endpoint_bot_engine_check
    assert "bot_engine" in data["checks"]
E   KeyError: 'checks'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
____________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_exchange_check _____________________________________________________
tests\unit\test_endpoints.py:309: in test_readiness_endpoint_exchange_check
    assert "exchange" in data["checks"]
E   KeyError: 'checks'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
______________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_cache_check ______________________________________________________
tests\unit\test_endpoints.py:325: in test_readiness_endpoint_cache_check
    assert "cache" in data["checks"]
E   KeyError: 'checks'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
__________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_message_queue_check __________________________________________________
tests\unit\test_endpoints.py:337: in test_readiness_endpoint_message_queue_check
    assert "message_queue" in data["checks"]
E   KeyError: 'checks'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
____________________________________________________ TestReadinessEndpoint.test_readiness_endpoint_database_check _____________________________________________________
tests\unit\test_endpoints.py:353: in test_readiness_endpoint_database_check
    assert "database" in data["checks"]
E   KeyError: 'checks'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/ready
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 412, in readiness_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/ready "HTTP/1.1 500 Internal Server Error"
______________________________________________________ TestDashboardEndpoint.test_dashboard_endpoint_returns_200 ______________________________________________________
tests\unit\test_endpoints.py:527: in test_dashboard_endpoint_returns_200
    assert response.status_code == 200
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    slowapi:extension.py:520 Skipping limit: 60 per 1 minute. Empty value found in parameters.
ERROR    api.app:app.py:275 Unhandled exception in GET /dashboard
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 478, in dashboard
    return templates.TemplateResponse(request, "dashboard.html")
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\templating.py", line 106, in TemplateResponse
    raise ValueError('context must include a "request" key')
ValueError: context must include a "request" key
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/dashboard "HTTP/1.1 500 Internal Server Error"
_____________________________________________________ TestDashboardEndpoint.test_dashboard_endpoint_returns_html ______________________________________________________
tests\unit\test_endpoints.py:532: in test_dashboard_endpoint_returns_html
    assert "text/html" in response.headers.get("content-type", "")
E   AssertionError: assert 'text/html' in 'application/json'
E    +  where 'application/json' = <bound method Headers.get of Headers({'content-length': '110', 'content-type': 'application/json'})>('content-type', '')
E    +    where <bound method Headers.get of Headers({'content-length': '110', 'content-type': 'application/json'})> = Headers({'content-length': '110', 'content-type': 'application/json'}).get
E    +      where Headers({'content-length': '110', 'content-type': 'application/json'}) = <Response [500 Internal Server Error]>.headers
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    slowapi:extension.py:520 Skipping limit: 60 per 1 minute. Empty value found in parameters.
ERROR    api.app:app.py:275 Unhandled exception in GET /dashboard
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 478, in dashboard
    return templates.TemplateResponse(request, "dashboard.html")
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\templating.py", line 106, in TemplateResponse
    raise ValueError('context must include a "request" key')
ValueError: context must include a "request" key
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/dashboard "HTTP/1.1 500 Internal Server Error"
__________________________________________________________ TestRateLimiting.test_rate_limit_headers_present ___________________________________________________________
tests\unit\test_endpoints.py:628: in test_rate_limit_headers_present
    assert response.status_code == 200
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    api.app:app.py:275 Unhandled exception in GET /api/v1/health
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 403, in health_check
    from core.healthcheck import get_health_check_manager
  File "C:\Users\TU\Desktop\new project\N1V1\core\healthcheck.py", line 17, in <module>
    import psycopg2
ModuleNotFoundError: No module named 'psycopg2'
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 500 Internal Server Error"
______________________________________________________ TestRateLimiting.test_dashboard_endpoint_not_rate_limited ______________________________________________________
tests\unit\test_endpoints.py:679: in test_dashboard_endpoint_not_rate_limited
    assert response.status_code == 200
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    slowapi:extension.py:520 Skipping limit: 60 per 1 minute. Empty value found in parameters.
ERROR    api.app:app.py:275 Unhandled exception in GET /dashboard
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 478, in dashboard
    return templates.TemplateResponse(request, "dashboard.html")
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\templating.py", line 106, in TemplateResponse
    raise ValueError('context must include a "request" key')
ValueError: context must include a "request" key
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/dashboard "HTTP/1.1 500 Internal Server Error"
________________________________________________________ TestCORSSecurity.test_cors_allows_configured_origins _________________________________________________________
tests\unit\test_endpoints.py:692: in test_cors_allows_configured_origins
    assert "access-control-allow-origin" in response.headers
E   AssertionError: assert 'access-control-allow-origin' in Headers({'content-length': '48', 'content-type': 'application/json', 'x-ratelimit-limit': '60', 'x-ratelimit-remaining': '0', 'x-ratelimit-reset': '1758596906.5483854', 'retry-after': '52'})
E    +  where Headers({'content-length': '48', 'content-type': 'application/json', 'x-ratelimit-limit': '60', 'x-ratelimit-remaining': '0', 'x-ratelimit-reset': '1758596906.5483854', 'retry-after': '52'}) = <Response [429 Too Many Requests]>.headers
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  slowapi:extension.py:510 ratelimit 60 per 1 minute (testclient) exceeded at endpoint: /api/v1/health
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 429 Too Many Requests"
__________________________________________ TestCustomExceptionMiddleware.test_custom_exception_middleware_handles_exceptions __________________________________________
tests\unit\test_endpoints.py:860: in test_custom_exception_middleware_handles_exceptions
    assert any('CustomExceptionMiddleware' in str(cls) for cls in middleware_classes)
E   assert False
E    +  where False = any(<generator object TestCustomExceptionMiddleware.test_custom_exception_middleware_handles_exceptions.<locals>.<genexpr> at 0x0000025808CD2F80>)
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________________ TestTemplateRendering.test_dashboard_template_not_found _______________________________________________________
tests\unit\test_endpoints.py:980: in test_dashboard_template_not_found
    assert response.status_code == 200
E   assert 500 == 200
E    +  where 500 = <Response [500 Internal Server Error]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    slowapi:extension.py:520 Skipping limit: 60 per 1 minute. Empty value found in parameters.
ERROR    api.app:app.py:275 Unhandled exception in GET /dashboard
Traceback (most recent call last):
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 50, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 79, in __call__
    raise exc
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\middleware\exceptions.py", line 68, in __call__
    await self.app(scope, receive, sender)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 20, in __call__
    raise e
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\middleware\asyncexitstack.py", line 17, in __call__
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 718, in __call__
    await route.handle(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 276, in handle
    await self.app(scope, receive, send)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\routing.py", line 66, in app
    response = await func(request)
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 274, in app
    raw_response = await run_endpoint_function(
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\fastapi\routing.py", line 191, in run_endpoint_function
    return await dependant.call(**values)
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\..\..\api\app.py", line 478, in dashboard
    return templates.TemplateResponse(request, "dashboard.html")
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\templating.py", line 106, in TemplateResponse
    raise ValueError('context must include a "request" key')
ValueError: context must include a "request" key
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/dashboard "HTTP/1.1 500 Internal Server Error"
_____________________________________________________ TestPrometheusMetrics.test_api_requests_counter_incremented _____________________________________________________
tests\unit\test_endpoints.py:991: in test_api_requests_counter_incremented
    assert initial_response.status_code == 200
E   assert 429 == 200
E    +  where 429 = <Response [429 Too Many Requests]>.status_code
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  slowapi:extension.py:510 ratelimit 60 per 1 minute (testclient) exceeded at endpoint: /api/v1/health
INFO     httpx:_client.py:1013 HTTP Request: GET http://testserver/api/v1/health "HTTP/1.1 429 Too Many Requests"
__________________________________________________ TestRateLimitingEdgeCases.test_get_remote_address_exempt_function __________________________________________________
tests\unit\test_endpoints.py:1097: in test_get_remote_address_exempt_function
    assert get_remote_address_exempt(non_exempt_request) is not None
api\app.py:117: in get_remote_address_exempt
    return get_remote_address(request)
venv\lib\site-packages\slowapi\util.py:24: in get_remote_address
    if not request.client or not request.client.host:
E   AttributeError: 'MockRequest' object has no attribute 'client'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________________ TestMiddlewareOrder.test_cors_middleware_configured _________________________________________________________
tests\unit\test_endpoints.py:1151: in test_cors_middleware_configured
    assert cors_found
E   assert False
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________________ TestMiddlewareOrder.test_rate_limit_middleware_configured ______________________________________________________
tests\unit\test_endpoints.py:1161: in test_rate_limit_middleware_configured
    assert rate_limit_found
E   assert False
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestStructuredLogging.test_logger_initialization ___________________________________________________________
tests\unit\test_logging_and_resources.py:27: in test_logger_initialization
    assert logger.sensitivity == LogSensitivity.INFO
E   AssertionError: assert <LogSensitivity.SECURE: 'secure'> == <LogSensitivity.INFO: 'info'>
E    +  where <LogSensitivity.SECURE: 'secure'> = <core.logging_utils.StructuredLogger object at 0x000002580859BD90>.sensitivity
E    +  and   <LogSensitivity.INFO: 'info'> = LogSensitivity.INFO
_______________________________________________________ TestCacheResourceManagement.test_cache_context_manager ________________________________________________________
venv\lib\site-packages\pytest_asyncio\plugin.py:440: in runtest
    super().runtest()
venv\lib\site-packages\pytest_asyncio\plugin.py:907: in inner
    _loop.run_until_complete(task)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1393: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________ TestCacheResourceManagement.test_cache_initialization_failure ____________________________________________________
venv\lib\site-packages\pytest_asyncio\plugin.py:440: in runtest
    super().runtest()
venv\lib\site-packages\pytest_asyncio\plugin.py:907: in inner
    _loop.run_until_complete(task)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1393: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________ TestCacheResourceManagement.test_cache_close_error_handling _____________________________________________________
venv\lib\site-packages\pytest_asyncio\plugin.py:440: in runtest
    super().runtest()
venv\lib\site-packages\pytest_asyncio\plugin.py:907: in inner
    _loop.run_until_complete(task)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1393: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_______________________________________________ TestResourceCleanupIntegration.test_cache_context_manager_with_cleanup ________________________________________________
venv\lib\site-packages\pytest_asyncio\plugin.py:440: in runtest
    super().runtest()
venv\lib\site-packages\pytest_asyncio\plugin.py:907: in inner
    _loop.run_until_complete(task)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1393: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________ TestResourceCleanupIntegration.test_cache_global_instance_cleanup __________________________________________________
venv\lib\site-packages\pytest_asyncio\plugin.py:440: in runtest
    super().runtest()
venv\lib\site-packages\pytest_asyncio\plugin.py:907: in inner
    _loop.run_until_complete(task)
..\..\..\AppData\Local\Programs\Python\Python310\lib\asyncio\base_events.py:649: in run_until_complete
    return future.result()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1393: in patched
    with self.decoration_helper(patched,
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:135: in __enter__
    return next(self.gen)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1358: in decoration_helper
    arg = exit_stack.enter_context(patching)
..\..\..\AppData\Local\Programs\Python\Python310\lib\contextlib.py:492: in enter_context
    result = _cm_type.__enter__(cm)
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1447: in __enter__
    original, local = self.get_original()
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:1420: in get_original
    raise AttributeError(
E   AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________ TestResourceCleanupIntegration.test_endpoint_global_instance_cleanup _________________________________________________
tests\unit\test_logging_and_resources.py:326: in test_endpoint_global_instance_cleanup
    await _cleanup_endpoint_on_exit()
E   TypeError: object NoneType can't be used in 'await' expression
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     metrics_endpoint:logging_utils.py:167 MetricsEndpoint initialized on localhost:9090/metrics | component=metrics_endpoint | operation=initialize | host=localhost | port=9090 | path=/metrics
INFO     metrics_endpoint:logging_utils.py:167 Metrics endpoint cleanup handler registered with atexit | component=metrics_endpoint | operation=initialize
INFO     metrics_endpoint:logging_utils.py:167 Performing metrics endpoint cleanup on application exit | component=metrics_endpoint | operation=cleanup
ERROR    metrics_endpoint:logging_utils.py:181 Error during metrics endpoint cleanup on exit: asyncio.run() cannot be called from a running event loop | component=metrics_endpoint | operation=cleanup | error=asyncio.run() cannot be called from a running event loop
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
___________________________________________________ TestResourceCleanupIntegration.test_cleanup_on_startup_failure ____________________________________________________
tests\unit\test_logging_and_resources.py:353: in test_cleanup_on_startup_failure
    assert endpoint.app is None
E   assert <Application 0x25806844430> is None
E    +  where <Application 0x25806844430> = <core.metrics_endpoint.MetricsEndpoint object at 0x00000258068452A0>.app
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     metrics_endpoint:logging_utils.py:167 MetricsEndpoint initialized on localhost:9090/metrics | component=metrics_endpoint | operation=initialize | host=localhost | port=9090 | path=/metrics
ERROR    metrics_endpoint:logging_utils.py:181 SSL certificate/key file not found: [Errno 2] No such file or directory | component=metrics_endpoint | operation=start | error=[Errno 2] No such file or directory
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________ TestTradingMetricsCollection.test_trading_metrics_collection _____________________________________________________
tests\unit\test_monitoring_observability.py:279: in test_trading_metrics_collection
    await collect_trading_metrics(self.collector)
core\metrics_collector.py:543: in collect_trading_metrics
    total_orders = await collector.get_metric_value("trading_orders_total", {"account": "main", "status": "filled"}) or 0
E   TypeError: object numpy.float64 can't be used in 'await' expression
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________________ TestAlertingSystem.test_alert_deduplication _____________________________________________________________
tests\unit\test_monitoring_observability.py:474: in test_alert_deduplication
    assert not triggered2  # Should be suppressed due to deduplication
E   assert not True
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________________ TestAlertingSystem.test_notification_delivery ____________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected '_send_discord_notification' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_monitoring_observability.py:498: in test_notification_delivery
    mock_discord.assert_called_once()
E   AssertionError: Expected '_send_discord_notification' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________________ TestOrderManager.test_init_paper_mode ________________________________________________________________
tests\unit\test_order_manager.py:114: in test_init_paper_mode
    assert om.live_executor is None
E   assert <core.order_manager.MockLiveExecutor object at 0x0000025808E83130> is None
E    +  where <core.order_manager.MockLiveExecutor object at 0x0000025808E83130> = <core.order_manager.OrderManager object at 0x0000025808E83160>.live_executor
___________________________________________________________ TestOrderManager.test_execute_order_paper_mode ____________________________________________________________
tests\unit\test_order_manager.py:143: in test_execute_order_paper_mode
    assert result is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:29.976915Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:29.937787+00:00) to int (1758596909937)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:29.937787+00:00) to int (1758596909937)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestOrderManager.test_execute_order_backtest_mode __________________________________________________________
tests\unit\test_order_manager.py:167: in test_execute_order_backtest_mode
    assert result is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:30.149846Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:30.108717+00:00) to int (1758596910108)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:30.108717+00:00) to int (1758596910108)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestOrderManager.test_execute_order_safe_mode_active _________________________________________________________
tests\unit\test_order_manager.py:187: in test_execute_order_safe_mode_active
    assert result is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:30.319078Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:30.282100+00:00) to int (1758596910282)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:30.282100+00:00) to int (1758596910282)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
______________________________________________________ TestOrderManager.test_execute_order_live_mode_with_retry _______________________________________________________
tests\unit\test_order_manager.py:213: in test_execute_order_live_mode_with_retry
    assert result is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:30.478987Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:30.441935+00:00) to int (1758596910441)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:30.441935+00:00) to int (1758596910441)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
____________________________________________________ TestOrderManager.test_execute_order_safe_mode_trigger_counter ____________________________________________________
tests\unit\test_order_manager.py:366: in test_execute_order_safe_mode_trigger_counter
    assert hasattr(om, '_safe_mode_triggers')
E   AssertionError: assert False
E    +  where False = hasattr(<core.order_manager.OrderManager object at 0x000002580B9B0BE0>, '_safe_mode_triggers')
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:31.462816Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:31.423656+00:00) to int (1758596911423)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:31.423656+00:00) to int (1758596911423)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestOrderManager.test_execute_order_unknown_mode ___________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:940: in assert_called_once_with
    raise AssertionError(msg)
E   AssertionError: Expected 'execute_paper_order' to be called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_order_manager.py:390: in test_execute_order_unknown_mode
    mock_executors['paper'].execute_paper_order.assert_called_once_with(signal)
E   AssertionError: Expected 'execute_paper_order' to be called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:31.645235Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:31.601113+00:00) to int (1758596911601)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:31.601113+00:00) to int (1758596911601)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________________ TestOrderManager.test_validate_order_payload_valid __________________________________________________________
core\order_manager.py:467: in _validate_order_payload
    jsonschema.validate(instance=signal_dict, schema=schema)
venv\lib\site-packages\jsonschema\validators.py:1307: in validate
    raise error
E   jsonschema.exceptions.ValidationError: None is not of type 'number'
E
E   Failed validating 'type' in schema['properties']['amount']:
E       {'minimum': 1e-08, 'type': 'number'}
E
E   On instance['amount']:
E       None

The above exception was the direct cause of the following exception:
tests\unit\test_order_manager.py:642: in test_validate_order_payload_valid
    om._validate_order_payload(valid_signal)
core\order_manager.py:477: in _validate_order_payload
    raise ValueError(f"Invalid order payload: {error_msg}") from e
E   ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: None is not of type 'number'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:34.434690+00:00) to int (1758596914434)
_________________________________________________ TestOrderManager.test_validate_order_payload_invalid_symbol_format __________________________________________________
core\order_manager.py:467: in _validate_order_payload
    jsonschema.validate(instance=signal_dict, schema=schema)
venv\lib\site-packages\jsonschema\validators.py:1307: in validate
    raise error
E   jsonschema.exceptions.ValidationError: None is not of type 'number'
E
E   Failed validating 'type' in schema['properties']['amount']:
E       {'minimum': 1e-08, 'type': 'number'}
E
E   On instance['amount']:
E       None

The above exception was the direct cause of the following exception:
tests\unit\test_order_manager.py:675: in test_validate_order_payload_invalid_symbol_format
    om._validate_order_payload(invalid_signal)
core\order_manager.py:477: in _validate_order_payload
    raise ValueError(f"Invalid order payload: {error_msg}") from e
E   ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'

During handling of the above exception, another exception occurred:
tests\unit\test_order_manager.py:674: in test_validate_order_payload_invalid_symbol_format
    with pytest.raises(ValueError, match="Symbol must be in format"):
E   AssertionError: Regex pattern did not match.
E    Regex: 'Symbol must be in format'
E    Input: "Invalid order payload: Schema validation failed: None is not of type 'number'"
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: None is not of type 'number'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:34.935898+00:00) to int (1758596914935)
_____________________________________________ TestOrderManager.test_validate_order_payload_business_rules_negative_amount _____________________________________________
core\order_manager.py:467: in _validate_order_payload
    jsonschema.validate(instance=signal_dict, schema=schema)
venv\lib\site-packages\jsonschema\validators.py:1307: in validate
    raise error
E   jsonschema.exceptions.ValidationError: None is not of type 'number'
E
E   Failed validating 'type' in schema['properties']['amount']:
E       {'minimum': 1e-08, 'type': 'number'}
E
E   On instance['amount']:
E       None

The above exception was the direct cause of the following exception:
tests\unit\test_order_manager.py:691: in test_validate_order_payload_business_rules_negative_amount
    om._validate_order_payload(invalid_signal)
core\order_manager.py:477: in _validate_order_payload
    raise ValueError(f"Invalid order payload: {error_msg}") from e
E   ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'

During handling of the above exception, another exception occurred:
tests\unit\test_order_manager.py:690: in test_validate_order_payload_business_rules_negative_amount
    with pytest.raises(ValueError, match="Order amount must be positive"):
E   AssertionError: Regex pattern did not match.
E    Regex: 'Order amount must be positive'
E    Input: "Invalid order payload: Schema validation failed: None is not of type 'number'"
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: None is not of type 'number'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:35.591890+00:00) to int (1758596915591)
____________________________________________ TestOrderManager.test_validate_order_payload_business_rules_stop_without_loss ____________________________________________
core\order_manager.py:467: in _validate_order_payload
    jsonschema.validate(instance=signal_dict, schema=schema)
venv\lib\site-packages\jsonschema\validators.py:1307: in validate
    raise error
E   jsonschema.exceptions.ValidationError: None is not of type 'number'
E
E   Failed validating 'type' in schema['properties']['amount']:
E       {'minimum': 1e-08, 'type': 'number'}
E
E   On instance['amount']:
E       None

The above exception was the direct cause of the following exception:
tests\unit\test_order_manager.py:707: in test_validate_order_payload_business_rules_stop_without_loss
    om._validate_order_payload(invalid_signal)
core\order_manager.py:477: in _validate_order_payload
    raise ValueError(f"Invalid order payload: {error_msg}") from e
E   ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'

During handling of the above exception, another exception occurred:
tests\unit\test_order_manager.py:706: in test_validate_order_payload_business_rules_stop_without_loss
    with pytest.raises(ValueError, match="Stop orders must include stop_loss"):
E   AssertionError: Regex pattern did not match.
E    Regex: 'Stop orders must include stop_loss'
E    Input: "Invalid order payload: Schema validation failed: None is not of type 'number'"
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: None is not of type 'number'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:35.974930+00:00) to int (1758596915974)
_______________________________________ TestOrderManager.test_validate_order_payload_business_rules_invalid_signal_order_combo ________________________________________
core\order_manager.py:467: in _validate_order_payload
    jsonschema.validate(instance=signal_dict, schema=schema)
venv\lib\site-packages\jsonschema\validators.py:1307: in validate
    raise error
E   jsonschema.exceptions.ValidationError: None is not of type 'number'
E
E   Failed validating 'type' in schema['properties']['amount']:
E       {'minimum': 1e-08, 'type': 'number'}
E
E   On instance['amount']:
E       None

The above exception was the direct cause of the following exception:
tests\unit\test_order_manager.py:723: in test_validate_order_payload_business_rules_invalid_signal_order_combo
    om._validate_order_payload(invalid_signal)
core\order_manager.py:477: in _validate_order_payload
    raise ValueError(f"Invalid order payload: {error_msg}") from e
E   ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'

During handling of the above exception, another exception occurred:
tests\unit\test_order_manager.py:722: in test_validate_order_payload_business_rules_invalid_signal_order_combo
    with pytest.raises(ValueError, match="Entry signals should use MARKET or LIMIT"):
E   AssertionError: Regex pattern did not match.
E    Regex: 'Entry signals should use MARKET or LIMIT'
E    Input: "Invalid order payload: Schema validation failed: None is not of type 'number'"
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: None is not of type 'number'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:36.328384+00:00) to int (1758596916328)
_______________________________________________________ TestOrderManager.test_execute_order_with_valid_payload ________________________________________________________
tests\unit\test_order_manager.py:762: in test_execute_order_with_valid_payload
    assert result is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:36.867737Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: None is not of type 'number'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: None is not of type 'number'
Order validation failed: Invalid order payload: Schema validation failed: None is not of type 'number'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:36.825760+00:00) to int (1758596916825)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:36.825760+00:00) to int (1758596916825)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________ TestPaperOrderExecutor.test_execute_paper_order_buy_signal ______________________________________________________
tests\unit\test_paper_executor.py:154: in test_execute_paper_order_buy_signal
    assert result.cost == Decimal("50025")  # 1.0 * 50000 with slippage
E   AssertionError: assert Decimal('50050.0000') == Decimal('50025')
E    +  where Decimal('50050.0000') = Order(id='paper_0', symbol='BTC/USDT', type=<OrderType.MARKET: 'market'>, side='buy', amount=Decimal('1.0'), price=Dec...00'), fee={'cost': 0.001, 'currency': 'USDT'}, trailing_stop=None, timestamp=1758571717349, params={'stop_loss': None}).cost
E    +  and   Decimal('50025') = Decimal('50025')
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________ TestPaperOrderExecutor.test_execute_paper_order_sell_signal _____________________________________________________
tests\unit\test_paper_executor.py:185: in test_execute_paper_order_sell_signal
    assert result.cost == Decimal("5997")  # 2.0 * 3000 with slippage
E   AssertionError: assert Decimal('5994.0000') == Decimal('5997')
E    +  where Decimal('5994.0000') = Order(id='paper_0', symbol='ETH/USDT', type=<OrderType.MARKET: 'market'>, side='sell', amount=Decimal('2.0'), price=De...00'), fee={'cost': 0.002, 'currency': 'USDT'}, trailing_stop=None, timestamp=1758571717419, params={'stop_loss': None}).cost
E    +  and   Decimal('5997') = Decimal('5997')
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestPaperOrderExecutor.test_execute_paper_order_portfolio_mode_buy __________________________________________________
tests\unit\test_paper_executor.py:308: in test_execute_paper_order_portfolio_mode_buy
    assert abs(paper_executor_with_balance.paper_balances["BTC/USDT"] - expected_btc_balance) < Decimal("0.1")
E   AssertionError: assert Decimal('1.200050') < Decimal('0.1')
E    +  where Decimal('1.200050') = abs((Decimal('497497.499950') - Decimal('497498.70')))
E    +  and   Decimal('0.1') = Decimal('0.1')
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
________________________________________________________ TestPaperOrderExecutor.test_apply_slippage_buy_order _________________________________________________________
tests\unit\test_paper_executor.py:414: in test_apply_slippage_buy_order
    assert adjusted_price == Decimal("50025")
E   AssertionError: assert Decimal('50050.000') == Decimal('50025')
E    +  where Decimal('50025') = Decimal('50025')
________________________________________________________ TestPaperOrderExecutor.test_apply_slippage_sell_order ________________________________________________________
tests\unit\test_paper_executor.py:424: in test_apply_slippage_sell_order
    assert adjusted_price == Decimal("49975")
E   AssertionError: assert Decimal('49950.000') == Decimal('49975')
E    +  where Decimal('49975') = Decimal('49975')
_____________________________________________________ TestPaperOrderExecutor.test_apply_slippage_with_dict_signal _____________________________________________________
tests\unit\test_paper_executor.py:433: in test_apply_slippage_with_dict_signal
    assert adjusted_price == Decimal("30015")
E   AssertionError: assert Decimal('30030.000') == Decimal('30015')
E    +  where Decimal('30015') = Decimal('30015')
________________________________________________ TestPaperOrderExecutor.test_execute_paper_order_different_order_types ________________________________________________
tests\unit\test_paper_executor.py:537: in test_execute_paper_order_different_order_types
    assert result.type == OrderType.LIMIT
E   AssertionError: assert <OrderType.MARKET: 'market'> == <OrderType.LIMIT: 'limit'>
E    +  where <OrderType.MARKET: 'market'> = Order(id='paper_0', symbol='BTC/USDT', type=<OrderType.MARKET: 'market'>, side='buy', amount=Decimal('1.0'), price=Dec...00'), fee={'cost': 0.001, 'currency': 'USDT'}, trailing_stop=None, timestamp=1758571717944, params={'stop_loss': None}).type
E    +  and   <OrderType.LIMIT: 'limit'> = OrderType.LIMIT
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestRegression.test_network_error_retry_mechanism __________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'retry_async' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_regression.py:129: in test_network_error_retry_mechanism
    mock_managers['reliability'].retry_async.assert_called_once()
E   AssertionError: Expected 'retry_async' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:42.357584Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:42.317714+00:00) to int (1758596922317)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:42.317714+00:00) to int (1758596922317)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________________ TestRegression.test_exchange_error_handling _____________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_regression.py:157: in test_exchange_error_handling
    mock_managers['reliability'].record_critical_error.assert_called_once()
E   AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:42.721236Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:42.681028+00:00) to int (1758596922681)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:42.681028+00:00) to int (1758596922681)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________________ TestRegression.test_timeout_error_recovery ______________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_regression.py:182: in test_timeout_error_recovery
    mock_managers['reliability'].record_critical_error.assert_called_once()
E   AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:08:43.085342Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:43.044567+00:00) to int (1758596923044)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:08:43.044567+00:00) to int (1758596923044)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________ TestRegression.test_memory_leak_prevention_in_long_running_sessions _________________________________________________
tests\unit\test_regression.py:382: in test_memory_leak_prevention_in_long_running_sessions
    assert all(result is not None for result in results)
E   assert False
E    +  where False = all(<generator object TestRegression.test_memory_leak_prevention_in_long_running_sessions.<locals>.<genexpr> at 0x000002580BA8B7D0>)
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:09:02.023180Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.062857Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.102457Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.145357Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.185273Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.221583Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.263895Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.304052Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.345012Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.383148Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.421673Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.467459Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.506141Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.545680Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.585781Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.622402Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.666350Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.703019Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.742973Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.785320Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.833626Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.874040Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.915017Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.954487Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:02.992832Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.034908Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.082179Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.119981Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.159949Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.202639Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.242975Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.279943Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.317934Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.356937Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.398078Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.441749Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.479699Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.524016Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.569025Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.607791Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.645099Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.697996Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.737430Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.787150Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.823716Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.871788Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.910686Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.952642Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:03.990187Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.028710Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.072639Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.109215Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.155244Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.193818Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.235640Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.273171Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.319837Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.357843Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.394088Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.435072Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.475222Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.511328Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.547293Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.586146Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.628723Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.668523Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.707546Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.756111Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.804486Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.843551Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.886135Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.923037Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:04.964396Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.012471Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.050560Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.098478Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.144017Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.191644Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.231446Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.282509Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.321662Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.361556Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.397328Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.439281Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.480270Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.612740Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.712564Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.780568Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.868718Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.933681Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:05.980801Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.026331Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.066128Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.104144Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.157986Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.198781Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.239744Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.278822Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.316602Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
{"timestamp": "2025-09-23T03:09:06.353870Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.981242+00:00) to int (1758596941981)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.982241+00:00) to int (1758596941982)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:01.983241+00:00) to int (1758596941983)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
__________________________________________________________ TestRegression.test_invalid_trading_mode_fallback __________________________________________________________
tests\unit\test_regression.py:424: in test_invalid_trading_mode_fallback
    assert result is not None
E   assert None is not None
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:09:06.558435Z", "level": "TRADE", "module": "crypto_bot", "message": "Order failed: validation_error: Invalid order payload: Schema validation failed: 1 is not of type 'string'", "correlation_id": null, "request_id": null, "strategy_id": null}
------------------------------------------------------------------------ Captured stderr call -------------------------------------------------------------------------
Order payload validation failed: Schema validation failed: 1 is not of type 'string'
Order validation failed: Invalid order payload: Schema validation failed: 1 is not of type 'string'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:06.516984+00:00) to int (1758596946516)
DEBUG    root:adapter.py:126 Normalized timestamp in dataclass from datetime (2025-09-23 03:09:06.516984+00:00) to int (1758596946516)
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_____________________________________________________________ TestLogSanitizer.test_api_key_sanitization ______________________________________________________________
tests\unit\test_secure_logging.py:36: in test_api_key_sanitization
    assert "***API_KEY_MASKED***" in sanitized
E   assert '***API_KEY_MASKED***' in 'Using API key: "sk-1234567890abcdef" for authentication'
_________________________________________________________ TestLogSanitizer.test_financial_amount_sanitization _________________________________________________________
tests\unit\test_secure_logging.py:52: in test_financial_amount_sanitization
    assert "***BALANCE_MASKED***" in sanitized
E   AssertionError: assert '***BALANCE_MASKED***' in '***12345.67_MASKED***, ***-987.65_MASKED***, ***11234.56_MASKED***'
__________________________________________________________ TestLogSanitizer.test_audit_level_minimal_output ___________________________________________________________
tests\unit\test_secure_logging.py:83: in test_audit_level_minimal_output
    assert sanitized == "[AUDIT] Security event logged"
E   AssertionError: assert 'Security eve...y sk-123 used' == '[AUDIT] Secu... event logged'
E
E     - [AUDIT] Security event logged
E     + Security event: API key sk-123 used
_________________________________________________________ TestLogSanitizer.test_complex_message_sanitization __________________________________________________________
tests\unit\test_secure_logging.py:97: in test_complex_message_sanitization
    assert "***API_KEY_MASKED***" in sanitized
E   assert '***API_KEY_MASKED***' in '\n        Processing trade for ***EMAIL_MASKED*** with API key "sk-1234567890abcdef"\n        ***50000.00_MASKED***, ***2500.75_MASKED***, Token: Bearer xyz789\n        Phone: +***PHONE_MASKED***, SSN: ***PHONE_MASKED***\n        '
__________________________________________________________ TestStructuredLogger.test_structured_info_logging __________________________________________________________
tests\unit\test_secure_logging.py:130: in test_structured_info_logging
    log_data = json.loads(log_output.split(" - ", 3)[-1])
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\__init__.py:346: in loads
    return _default_decoder.decode(s)
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\decoder.py:337: in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\decoder.py:355: in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
E   json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     test_logger:logging_utils.py:167 Test message | symbol=BTC/USDT | amount=1000.5 | user_id=user123
________________________________________________________ TestStructuredLogger.test_structured_warning_logging _________________________________________________________
tests\unit\test_secure_logging.py:142: in test_structured_warning_logging
    assert "WARNING" in log_output
E   AssertionError: assert 'WARNING' in 'Alert triggered | alert_type=high_cpu | threshold=90.5\n'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
WARNING  test_logger:logging_utils.py:174 Alert triggered | alert_type=high_cpu | threshold=90.5
_________________________________________________________ TestStructuredLogger.test_structured_error_logging __________________________________________________________
tests\unit\test_secure_logging.py:150: in test_structured_error_logging
    assert "ERROR" in log_output
E   AssertionError: assert 'ERROR' in 'Database connection failed | db_host=localhost | error_code=500\n'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
ERROR    test_logger:logging_utils.py:181 Database connection failed | db_host=localhost | error_code=500
________________________________________________________ TestStructuredLogger.test_debug_level_preserves_data _________________________________________________________
tests\unit\test_secure_logging.py:164: in test_debug_level_preserves_data
    log_data = json.loads(log_output.split(" - ", 3)[-1])
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\__init__.py:346: in loads
    return _default_decoder.decode(s)
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\decoder.py:337: in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\decoder.py:355: in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
E   json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     debug_logger:logging_utils.py:167 Debug message | api_key=sk-test123 | balance=1000.5
____________________________________________________________ TestStructuredLogger.test_sensitivity_change _____________________________________________________________
tests\unit\test_secure_logging.py:184: in test_sensitivity_change
    assert "***API_KEY_MASKED***" in secure_output
E   AssertionError: assert '***API_KEY_MASKED***' in 'Test message | api_key=sk-test123\n'
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     test_logger:logging_utils.py:167 Test message | api_key=sk-test123
INFO     test_logger:logging_utils.py:167 Test message | api_key=sk-test123
_______________________________________________________ TestCoreLoggerIntegration.test_core_logger_sanitization _______________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'info' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_secure_logging.py:208: in test_core_logger_sanitization
    mock_info.assert_called_once()
E   AssertionError: Expected 'info' to have been called once. Called 0 times.
________________________________________________________ TestLogSecurityVerification.test_no_api_keys_in_logs _________________________________________________________
tests\unit\test_secure_logging.py:267: in test_no_api_keys_in_logs
    assert api_key not in sanitized
E   AssertionError: assert 'sk-1234567890abcdef' not in 'Using API k...567890abcdef'
E
E     'sk-1234567890abcdef' is contained here:
E       Using API key: sk-1234567890abcdef
____________________________________________________ TestLogSecurityVerification.test_no_financial_amounts_in_logs ____________________________________________________
tests\unit\test_secure_logging.py:285: in test_no_financial_amounts_in_logs
    assert amount not in sanitized
E   AssertionError: assert '12345.67' not in '***12345.67_MASKED***'
E
E     '12345.67' is contained here:
E       ***12345.67_MASKED***
E     ?    ++++++++
_____________________________________________________ TestLogSecurityVerification.test_log_structure_preservation _____________________________________________________
tests\unit\test_secure_logging.py:320: in test_log_structure_preservation
    assert "***AMOUNT_MASKED***" in sanitized
E   AssertionError: assert '***AMOUNT_MASKED***' in 'Processing trade for ***EMAIL_MASKED*** with amount 12345.67 and API key sk-123'
____________________________________________________________ TestJournalWriter.test_append_with_event_loop ____________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:908: in assert_called_once
    raise AssertionError(msg)
E   AssertionError: Expected 'create_task' to have been called once. Called 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_signal_router.py:98: in test_append_with_event_loop
    self.task_manager.create_task.assert_called_once()
E   AssertionError: Expected 'create_task' to have been called once. Called 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________________________ TestJournalWriter.test_stop_method __________________________________________________________________
..\..\..\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py:2277: in assert_awaited_once
    raise AssertionError(msg)
E   AssertionError: Expected mock to have been awaited once. Awaited 0 times.

During handling of the above exception, another exception occurred:
tests\unit\test_signal_router.py:112: in test_stop_method
    mock_task.assert_awaited_once()
E   AssertionError: Expected mock to have been awaited once. Awaited 0 times.
------------------------------------------------------------------------- Captured log setup --------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
------------------------------------------------------------------------ Captured log teardown ------------------------------------------------------------------------
DEBUG    asyncio:proactor_events.py:630 Using proactor: IocpProactor
_________________________________________________________ TestJSONFormatter.test_format_record_with_exception _________________________________________________________
tests\utils\test_logger.py:860: in test_format_record_with_exception
    raise ValueError("Test exception")
E   ValueError: Test exception

During handling of the above exception, another exception occurred:
tests\utils\test_logger.py:862: in test_format_record_with_exception
    exc_info = sys.exc_info()
E   NameError: name 'sys' is not defined
________________________________________________ TestEnvironmentVariables.test_setup_logging_with_env_log_format_json _________________________________________________
tests\utils\test_logger.py:994: in test_setup_logging_with_env_log_format_json
    assert len(console_handlers) == 1
E   assert 2 == 1
E    +  where 2 = len([<StreamHandler <tempfile._TemporaryFileWrapper object at 0x000002585CF723E0> (INFO)>, <RotatingFileHandler C:\Users\TU\Desktop\new project\N1V1\logs\crypto_bot.log (INFO)>])
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     utils.logger:logger.py:782 TradeLogger subscribed to event bus for event-driven logging
_______________________________________________ TestEnvironmentVariables.test_setup_logging_with_env_log_format_pretty ________________________________________________
tests\utils\test_logger.py:1005: in test_setup_logging_with_env_log_format_pretty
    assert len(console_handlers) == 1
E   assert 2 == 1
E    +  where 2 = len([<StreamHandler <tempfile._TemporaryFileWrapper object at 0x000002585CF723E0> (INFO)>, <RotatingFileHandler C:\Users\TU\Desktop\new project\N1V1\logs\crypto_bot.log (INFO)>])
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     utils.logger:logger.py:782 TradeLogger subscribed to event bus for event-driven logging
________________________________________________ TestEnvironmentVariables.test_setup_logging_with_env_log_format_color ________________________________________________
tests\utils\test_logger.py:1016: in test_setup_logging_with_env_log_format_color
    assert len(console_handlers) == 1
E   assert 2 == 1
E    +  where 2 = len([<StreamHandler <tempfile._TemporaryFileWrapper object at 0x000002585CF723E0> (INFO)>, <RotatingFileHandler C:\Users\TU\Desktop\new project\N1V1\logs\crypto_bot.log (INFO)>])
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     utils.logger:logger.py:782 TradeLogger subscribed to event bus for event-driven logging
______________________________________________________ TestCorrelationIdSupport.test_generate_request_id_format _______________________________________________________
tests\utils\test_logger.py:1029: in test_generate_request_id_format
    assert len(request_id) == 21  # "req_" + 16 hex chars
E   AssertionError: assert 20 == 21
E    +  where 20 = len('req_3446f01108aa444f')
______________________________________________________ TestLoggerIntegration.test_structured_logging_integration ______________________________________________________
tests\utils\test_logger.py:1217: in test_structured_logging_integration
    parsed = json.loads(output.strip())
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\__init__.py:346: in loads
    return _default_decoder.decode(s)
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\decoder.py:337: in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
..\..\..\AppData\Local\Programs\Python\Python310\lib\json\decoder.py:355: in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
E   json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
------------------------------------------------------------------------ Captured stdout call -------------------------------------------------------------------------
{"timestamp": "2025-09-23T03:10:21.783658Z", "level": "INFO", "module": "crypto_bot", "message": "Order execution started", "correlation_id": "corr_123", "request_id": "req_456", "strategy_id": "momentum_v1", "symbol": "BTC/USDT", "component": "order_executor"}
-------------------------------------------------------------------------- Captured log call --------------------------------------------------------------------------
INFO     utils.logger:logger.py:782 TradeLogger subscribed to event bus for event-driven logging
========================================================================== warnings summary ===========================================================================
venv\lib\site-packages\starlette\formparsers.py:10
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\formparsers.py:10: PendingDeprecationWarning: Please use `import python_multipart` instead.
    import multipart

venv\lib\site-packages\pydantic\_internal\_fields.py:149
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pydantic\_internal\_fields.py:149: UserWarning: Field "model_version" has conflict with protected namespace "model_".

  You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    warnings.warn(

demo/test_simple_metrics.py::test_metrics_calculation
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py:197: PytestReturnNotNoneWarning: Expected None, but demo/test_simple_metrics.py::test_metrics_calculation returned True, which will be an error in a future version of pytest.  Did you mean to use `assert` instead of `return`?
    warnings.warn(

demo/test_simple_metrics.py::test_metrics_result_structure
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py:197: PytestReturnNotNoneWarning: Expected None, but demo/test_simple_metrics.py::test_metrics_result_structure returned False, which will be an error in a future version of pytest.  Did you mean to use `assert` instead of `return`?
    warnings.warn(

scripts/test_binary_labels.py::test_binary_labels
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py:197: PytestReturnNotNoneWarning: Expected None, but scripts/test_binary_labels.py::test_binary_labels returned True, which will be an error in a future version of pytest.  Did you mean to use `assert` instead of `return`?
    warnings.warn(

tests/data/test_data_fixes.py: 1 warning
tests/data/test_data_module_refactoring.py: 23 warnings
tests/data/test_historical_loader.py: 5 warnings
  C:\Users\TU\Desktop\new project\N1V1\data\historical_loader.py:502: FutureWarning: 'H' is deprecated and will be removed in a future version. Please use 'h' instead of 'H'.
    current_end = min(current_start + pd.Timedelta(self._get_pandas_freq(timeframe)), final_end)

tests/data/test_data_fixes.py::TestStructuredLogging::test_historical_loader_structured_logging
tests/data/test_data_module_refactoring.py::TestDataValidation::test_validate_data_valid
tests/data/test_historical_loader.py::TestHistoricalDataLoaderDataProcessing::test_validate_data_success
tests/data/test_historical_loader.py::TestHistoricalDataLoaderDataProcessing::test_validate_data_timeframe_consistency
  C:\Users\TU\Desktop\new project\N1V1\data\historical_loader.py:742: FutureWarning: 'H' is deprecated and will be removed in a future version. Please use 'h' instead of 'H'.
    expected_diff = pd.Timedelta(self._get_pandas_freq(timeframe))

tests/data/test_historical_loader.py::TestHistoricalDataLoaderResampling::test_resample_data_success
  C:\Users\TU\Desktop\new project\N1V1\data\historical_loader.py:1038: FutureWarning: 'H' is deprecated and will be removed in a future version, please use 'h' instead.
    resampled_df = df.resample(freq).agg(resample_map)

tests/integration/test_ml_serving_integration.py: 76 warnings
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\sklearn\base.py:458: UserWarning: X has feature names, but RandomForestClassifier was fitted without feature names
    warnings.warn(

tests/ml/test_ml.py::TestConfiguration::test_calculate_all_indicators
tests/ml/test_ml.py::TestFeatureExtractor::test_extract_features
tests/ml/test_ml.py::TestFeatureExtractor::test_get_feature_importance_template
tests/ml/test_ml.py::TestFeatureExtractor::test_get_feature_stats
  C:\Users\TU\Desktop\new project\N1V1\ml\indicators.py:249: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '1103.1238688359326' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
    obv.iloc[0] = data['volume'].iloc[0]

tests/ml/test_ml.py::TestIntegration::test_full_pipeline
  C:\Users\TU\Desktop\new project\N1V1\ml\indicators.py:249: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '1707.2386343133985' has dtype incompatible with int64, please explicitly cast to a compatible dtype first.
    obv.iloc[0] = data['volume'].iloc[0]

tests/strategies/test_strategy_integration.py: 15 warnings
  C:\Users\TU\Desktop\new project\N1V1\strategies\base_strategy.py:211: SettingWithCopyWarning:
  A value is trying to be set on a copy of a slice from a DataFrame.
  Try using .loc[row_indexer,col_indexer] = value instead

  See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    data["symbol"] = symbol  # Add symbol column

tests/utils/test_circular_import_fix.py::test_import_chain
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py:197: PytestReturnNotNoneWarning: Expected None, but tests/utils/test_circular_import_fix.py::test_import_chain returned False, which will be an error in a future version of pytest.  Did you mean to use `assert` instead of `return`?
    warnings.warn(

tests/utils/test_circular_import_fix.py::test_lazy_import_functionality
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py:197: PytestReturnNotNoneWarning: Expected None, but tests/utils/test_circular_import_fix.py::test_lazy_import_functionality returned False, which will be an error in a future version of pytest.  Did you mean to use `assert` instead of `return`?
    warnings.warn(

tests/utils/test_circular_import_fix.py::test_module_interactions
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\_pytest\python.py:197: PytestReturnNotNoneWarning: Expected None, but tests/utils/test_circular_import_fix.py::test_module_interactions returned True, which will be an error in a future version of pytest.  Did you mean to use `assert` instead of `return`?
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================================================================= short test summary info =======================================================================
FAILED tests/ml/test_ml_filter.py::TestMLFilter::test_save_load_model - assert False
FAILED tests/ml/test_ml_filter.py::TestFactoryFunctions::test_load_ml_filter - assert False
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_python_random - xgboost.core.XGBoostError: [10:03:54] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_numpy - xgboost.core.XGBoostError: [10:03:54] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_pandas - xgboost.core.XGBoostError: [10:03:55] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_lightgbm - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'lgb'
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_xgboost - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'xgb'
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_catboost - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'cb'
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_tensorflow - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'tf'
FAILED tests/ml/test_reproducibility.py::TestDeterministicSeeds::test_set_deterministic_seeds_pytorch - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'torch'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_packages - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'pkg_resources'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_env_vars - AssertionError: assert '/usr/local/bin:/usr/bin' == '/usr/local/lib/python3.9'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_git_info - KeyError: 'commit_hash'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_git_failure - AssertionError: assert 'error' in {}
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_hardware - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'psutil'
FAILED tests/ml/test_reproducibility.py::TestEnvironmentSnapshot::test_capture_environment_snapshot_hardware_failure - AttributeError: <module 'ml.train' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\ml\\train.py'> does not have the attribute 'psutil'
FAILED tests/ml/test_reproducibility.py::TestExperimentTrackerReproducibility::test_experiment_tracker_git_info_logging - KeyError: 'commit_hash'
FAILED tests/ml/test_reproducibility.py::TestReproducibilityValidation::test_deterministic_numpy_operations - xgboost.core.XGBoostError: [10:03:58] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityValidation::test_deterministic_pandas_operations - xgboost.core.XGBoostError: [10:03:58] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_full_reproducibility_workflow - xgboost.core.XGBoostError: [10:03:59] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_reproducibility_with_different_seeds - xgboost.core.XGBoostError: [10:03:59] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_reproducibility.py::TestReproducibilityIntegration::test_reproducibility_with_same_seed - xgboost.core.XGBoostError: [10:04:00] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_serving.py::TestPredictionProcessing::test_process_single_prediction_success - TypeError: argument of type 'coroutine' is not iterable
FAILED tests/ml/test_serving.py::TestPredictionProcessing::test_process_single_prediction_model_error - Failed: DID NOT RAISE <class 'Exception'>
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_success - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_with_nan_values - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_with_zero_prices - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestPrepareTrainingData::test_prepare_data_outlier_removal - KeyError: 'Open'
FAILED tests/ml/test_train.py::TestMainFunction::test_main_success - AssertionError: Expected 'exit' to not have been called. Called 1 times.
FAILED tests/ml/test_train.py::TestMainFunction::test_main_predictive_models_disabled - AssertionError: Expected 'exit' to not have been called. Called 1 times.
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_basic - xgboost.core.XGBoostError: [10:04:04] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_with_libs - xgboost.core.XGBoostError: [10:04:04] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
FAILED tests/ml/test_train.py::TestReproducibility::test_set_deterministic_seeds_libs_unavailable - xgboost.core.XGBoostError: [10:04:04] C:\buildkite-agent\builds\buildkite-windows-cpu-autoscaling-group-i-0750514818a16474a-1\xgboost\xgboost-ci-windows\src\c_api\...
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
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_normal - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required columns or invalid data
FAILED tests/risk/test_adaptive_policy.py::TestMarketConditionMonitor::test_assess_market_conditions_high_volatility - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required columns or invalid data
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_empty_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_none_market_data - assert 0.25 == 1.0
FAILED tests/risk/test_adaptive_policy.py::TestEdgeCases::test_market_monitor_error_handling - ValueError: Critical market data validation error for BTC/USDT: Market data validation failed - missing required columns or invalid data
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_calculation - assert False
FAILED tests/risk/test_adaptive_risk.py::TestAdaptiveRiskManagement::test_adaptive_position_size_fallback - AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x0000025806995FC0> == Decimal('200.00')
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
FAILED tests/risk/test_risk_manager_integration.py::TestRiskManagerAdaptiveIntegration::test_adaptive_atr_position_sizing_with_multiplier - AssertionError: assert <coroutine object RiskManager._calculate_adaptive_position_size_protected at 0x0000025806AABDF0> == Decimal('1.0')
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
FAILED tests/unit/test_binary_model_metrics.py::TestErrorHandling::test_collect_metrics_with_binary_integration_import_error - AttributeError: <module 'core.binary_model_metrics' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\binary_model_metrics.py'> does not have the attribute 'g...
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
FAILED tests/unit/test_cache_eviction.py::TestCacheIntegration::test_memory_manager_integration - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
FAILED tests/unit/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_valid_config - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/unit/test_config_manager.py::TestConfigManagerValidation::test_config_manager_init_with_missing_sections - utils.security.SecurityException: Required secret 'exchange_api_key' not found in secure storage
FAILED tests/unit/test_core_security.py::TestDataProcessorSecurity::test_calculate_rsi_batch_data_validation - AssertionError: assert 'TEST' not in {'TEST':    open\n0   100\n1   101\n2   102}
FAILED tests/unit/test_core_security.py::TestMetricsEndpointSecurity::test_secure_defaults - assert False == True
FAILED tests/unit/test_cross_feature_integration.py::TestCircuitBreakerMonitoringIntegration::test_monitoring_performance_during_circuit_breaker - AssertionError: Performance degraded by 54.4% during circuit breaker (threshold: 50%)
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_component_factory_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_cache_component_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_memory_manager_creation - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_component_caching - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_configuration_override - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_async_component_operations - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestDependencyInjection::test_factory_global_instance - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
FAILED tests/unit/test_dependency_injection.py::TestConfigurationIntegration::test_performance_tracker_config_integration - assert 1000.0 == 2000.0
FAILED tests/unit/test_dependency_injection.py::TestConfigurationPersistence::test_config_save_load - AttributeError: 'dict' object has no attribute 'market_ticker'
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_returns_200 - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_returns_json - AssertionError: assert 'status' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/health'}, 'message': 'An unexpected error occurred'}}
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_returns_metadata - KeyError: 'version'
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_correlation_id_unique - KeyError: 'correlation_id'
FAILED tests/unit/test_endpoints.py::TestHealthEndpoint::test_health_endpoint_timestamp_format - KeyError: 'timestamp'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_returns_200_when_healthy - assert 500 in [200, 503]
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_returns_json_structure - AssertionError: assert 'ready' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_includes_all_check_components - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_correlation_id_unique - KeyError: 'correlation_id'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_timestamp_format - KeyError: 'timestamp'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_returns_503_when_bot_engine_unavailable - assert 500 == 503
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_check_details_structure - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_latency_measurement - KeyError: 'total_latency_ms'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[DATABASE_URL] - AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[EXCHANGE_API_URL] - AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[MESSAGE_QUEUE_URL] - AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_handles_missing_env_vars[REDIS_URL] - AssertionError: assert 'checks' in {'error': {'code': 500, 'details': {'method': 'GET', 'path': '/api/v1/ready'}, 'message': 'An unexpected error occurred'}}
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_bot_engine_check - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_exchange_check - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_cache_check - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_message_queue_check - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestReadinessEndpoint::test_readiness_endpoint_database_check - KeyError: 'checks'
FAILED tests/unit/test_endpoints.py::TestDashboardEndpoint::test_dashboard_endpoint_returns_200 - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestDashboardEndpoint::test_dashboard_endpoint_returns_html - AssertionError: assert 'text/html' in 'application/json'
FAILED tests/unit/test_endpoints.py::TestRateLimiting::test_rate_limit_headers_present - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestRateLimiting::test_dashboard_endpoint_not_rate_limited - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestCORSSecurity::test_cors_allows_configured_origins - AssertionError: assert 'access-control-allow-origin' in Headers({'content-length': '48', 'content-type': 'application/json', 'x-ratelimit-limit': '60', 'x-ratelimi...
FAILED tests/unit/test_endpoints.py::TestCustomExceptionMiddleware::test_custom_exception_middleware_handles_exceptions - assert False
FAILED tests/unit/test_endpoints.py::TestTemplateRendering::test_dashboard_template_not_found - assert 500 == 200
FAILED tests/unit/test_endpoints.py::TestPrometheusMetrics::test_api_requests_counter_incremented - assert 429 == 200
FAILED tests/unit/test_endpoints.py::TestRateLimitingEdgeCases::test_get_remote_address_exempt_function - AttributeError: 'MockRequest' object has no attribute 'client'
FAILED tests/unit/test_endpoints.py::TestMiddlewareOrder::test_cors_middleware_configured - assert False
FAILED tests/unit/test_endpoints.py::TestMiddlewareOrder::test_rate_limit_middleware_configured - assert False
FAILED tests/unit/test_logging_and_resources.py::TestStructuredLogging::test_logger_initialization - AssertionError: assert <LogSensitivity.SECURE: 'secure'> == <LogSensitivity.INFO: 'info'>
FAILED tests/unit/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_context_manager - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
FAILED tests/unit/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_initialization_failure - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
FAILED tests/unit/test_logging_and_resources.py::TestCacheResourceManagement::test_cache_close_error_handling - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_context_manager_with_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cache_global_instance_cleanup - AttributeError: <module 'core.cache' from 'C:\\Users\\TU\\Desktop\\new project\\N1V1\\core\\cache.py'> does not have the attribute 'redis'
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_endpoint_global_instance_cleanup - TypeError: object NoneType can't be used in 'await' expression
FAILED tests/unit/test_logging_and_resources.py::TestResourceCleanupIntegration::test_cleanup_on_startup_failure - assert <Application 0x25806844430> is None
FAILED tests/unit/test_monitoring_observability.py::TestTradingMetricsCollection::test_trading_metrics_collection - TypeError: object numpy.float64 can't be used in 'await' expression
FAILED tests/unit/test_monitoring_observability.py::TestAlertingSystem::test_alert_deduplication - assert not True
FAILED tests/unit/test_monitoring_observability.py::TestAlertingSystem::test_notification_delivery - AssertionError: Expected '_send_discord_notification' to have been called once. Called 0 times.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_init_paper_mode - assert <core.order_manager.MockLiveExecutor object at 0x0000025808E83130> is None
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_paper_mode - assert None is not None
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_backtest_mode - assert None is not None
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_active - assert None is not None
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_live_mode_with_retry - assert None is not None
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_safe_mode_trigger_counter - AssertionError: assert False
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_unknown_mode - AssertionError: Expected 'execute_paper_order' to be called once. Called 0 times.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_valid - ValueError: Invalid order payload: Schema validation failed: None is not of type 'number'
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_invalid_symbol_format - AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_negative_amount - AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_stop_without_loss - AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_validate_order_payload_business_rules_invalid_signal_order_combo - AssertionError: Regex pattern did not match.
FAILED tests/unit/test_order_manager.py::TestOrderManager::test_execute_order_with_valid_payload - assert None is not None
FAILED tests/unit/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_buy_signal - AssertionError: assert Decimal('50050.0000') == Decimal('50025')
FAILED tests/unit/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_sell_signal - AssertionError: assert Decimal('5994.0000') == Decimal('5997')
FAILED tests/unit/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_portfolio_mode_buy - AssertionError: assert Decimal('1.200050') < Decimal('0.1')
FAILED tests/unit/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_buy_order - AssertionError: assert Decimal('50050.000') == Decimal('50025')
FAILED tests/unit/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_sell_order - AssertionError: assert Decimal('49950.000') == Decimal('49975')
FAILED tests/unit/test_paper_executor.py::TestPaperOrderExecutor::test_apply_slippage_with_dict_signal - AssertionError: assert Decimal('30030.000') == Decimal('30015')
FAILED tests/unit/test_paper_executor.py::TestPaperOrderExecutor::test_execute_paper_order_different_order_types - AssertionError: assert <OrderType.MARKET: 'market'> == <OrderType.LIMIT: 'limit'>
FAILED tests/unit/test_regression.py::TestRegression::test_network_error_retry_mechanism - AssertionError: Expected 'retry_async' to have been called once. Called 0 times.
FAILED tests/unit/test_regression.py::TestRegression::test_exchange_error_handling - AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
FAILED tests/unit/test_regression.py::TestRegression::test_timeout_error_recovery - AssertionError: Expected 'record_critical_error' to have been called once. Called 0 times.
FAILED tests/unit/test_regression.py::TestRegression::test_memory_leak_prevention_in_long_running_sessions - assert False
FAILED tests/unit/test_regression.py::TestRegression::test_invalid_trading_mode_fallback - assert None is not None
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_api_key_sanitization - assert '***API_KEY_MASKED***' in 'Using API key: "sk-1234567890abcdef" for authentication'
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_financial_amount_sanitization - AssertionError: assert '***BALANCE_MASKED***' in '***12345.67_MASKED***, ***-987.65_MASKED***, ***11234.56_MASKED***'
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_audit_level_minimal_output - AssertionError: assert 'Security eve...y sk-123 used' == '[AUDIT] Secu... event logged'
FAILED tests/unit/test_secure_logging.py::TestLogSanitizer::test_complex_message_sanitization - assert '***API_KEY_MASKED***' in '\n        Processing trade for ***EMAIL_MASKED*** with API key "sk-1234567890abcdef"\n        ***50000.00_MASKED***, ***2500.75_M...
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_structured_info_logging - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_structured_warning_logging - AssertionError: assert 'WARNING' in 'Alert triggered | alert_type=high_cpu | threshold=90.5\n'
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_structured_error_logging - AssertionError: assert 'ERROR' in 'Database connection failed | db_host=localhost | error_code=500\n'
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_debug_level_preserves_data - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
FAILED tests/unit/test_secure_logging.py::TestStructuredLogger::test_sensitivity_change - AssertionError: assert '***API_KEY_MASKED***' in 'Test message | api_key=sk-test123\n'
FAILED tests/unit/test_secure_logging.py::TestCoreLoggerIntegration::test_core_logger_sanitization - AssertionError: Expected 'info' to have been called once. Called 0 times.
FAILED tests/unit/test_secure_logging.py::TestLogSecurityVerification::test_no_api_keys_in_logs - AssertionError: assert 'sk-1234567890abcdef' not in 'Using API k...567890abcdef'
FAILED tests/unit/test_secure_logging.py::TestLogSecurityVerification::test_no_financial_amounts_in_logs - AssertionError: assert '12345.67' not in '***12345.67_MASKED***'
FAILED tests/unit/test_secure_logging.py::TestLogSecurityVerification::test_log_structure_preservation - AssertionError: assert '***AMOUNT_MASKED***' in 'Processing trade for ***EMAIL_MASKED*** with amount 12345.67 and API key sk-123'
FAILED tests/unit/test_signal_router.py::TestJournalWriter::test_append_with_event_loop - AssertionError: Expected 'create_task' to have been called once. Called 0 times.
FAILED tests/unit/test_signal_router.py::TestJournalWriter::test_stop_method - AssertionError: Expected mock to have been awaited once. Awaited 0 times.
FAILED tests/utils/test_logger.py::TestJSONFormatter::test_format_record_with_exception - NameError: name 'sys' is not defined
FAILED tests/utils/test_logger.py::TestEnvironmentVariables::test_setup_logging_with_env_log_format_json - assert 2 == 1
FAILED tests/utils/test_logger.py::TestEnvironmentVariables::test_setup_logging_with_env_log_format_pretty - assert 2 == 1
FAILED tests/utils/test_logger.py::TestEnvironmentVariables::test_setup_logging_with_env_log_format_color - assert 2 == 1
FAILED tests/utils/test_logger.py::TestCorrelationIdSupport::test_generate_request_id_format - AssertionError: assert 20 == 21
FAILED tests/utils/test_logger.py::TestLoggerIntegration::test_structured_logging_integration - json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_get_cache_key - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_empty - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_and_load_from_cache - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_save_to_cache_empty_dataframe - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherCaching::test_load_from_cache_expired - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_full_workflow_with_caching - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_multiple_symbols_workflow - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_data_fetcher.py::TestDataFetcherIntegration::test_realtime_data_workflow - data.data_fetcher.PathTraversalError: Invalid cache directory path
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_full_data_loading_workflow - data.historical_loader.ConfigurationError: Absolute path detected
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_data_validation_workflow - data.historical_loader.ConfigurationError: Absolute path detected
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_cache_key_generation_workflow - data.historical_loader.ConfigurationError: Absolute path detected
ERROR tests/data/test_historical_loader.py::TestHistoricalDataLoaderIntegration::test_timeframe_utilities_workflow - data.historical_loader.ConfigurationError: Absolute path detected
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_model_monitor_initialization - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_update_predictions - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_calculate_performance_metrics - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_detect_drift_no_reference - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_detect_drift_with_reference - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_check_model_health - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_trigger_alert - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_save_monitoring_data - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_generate_report - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_start_stop_monitoring - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/ml/test_model_monitor.py::TestModelMonitor::test_monitoring_loop - _pickle.PicklingError: Can't pickle <class 'unittest.mock.MagicMock'>: it's not the same object as unittest.mock.MagicMock
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_get_secret_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_get_secret_not_found - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_store_secret_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_rotate_key - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/security/test_key_management.py::TestVaultKeyManager::test_health_check_success - TypeError: VaultKeyManager.__init__() got an unexpected keyword argument 'url'
ERROR tests/test_integration.py::TestOptimizationIntegration::test_end_to_end_optimization_workflow - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_realtime_data, shutdown
ERROR tests/test_integration.py::TestOptimizationIntegration::test_component_interaction_data_to_backtest - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_realtime_data, shutdown
ERROR tests/test_integration.py::TestOptimizationIntegration::test_error_scenario_data_loading_failure - TypeError: Can't instantiate abstract class MockDataFetcher with abstract methods get_multiple_historical_data, get_realtime_data, shutdown
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_constant_returns - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_sharpe_ratio_single_return - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_profit_factor_edge_cases - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
ERROR tests/unit/test_algorithmic_correctness.py::TestPerformanceTrackerCorrectness::test_total_return_percentage_safe_division - ValueError: Configuration validation failed: Field 'trading': Extra inputs are not permitted; Field 'order': Extra inputs are not permitted; Field 'risk': Extra in...
ERROR tests/unit/test_cache_eviction.py::TestCacheIntegration::test_end_to_end_eviction_workflow
========================================== 315 failed, 2653 passed, 12 skipped, 138 warnings, 36 errors in 704.37s (0:11:44) ==========================================