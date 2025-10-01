(venv) C:\Users\TU\Desktop\new project\N1V1>pytest -q --tb=no
................................................................................................................................................................ [  5%]
................................................................................................................................................................ [ 10%]
................................................................................................................................................................ [ 15%]
................................................................................................................................................................ [ 20%]
................................................................................................................................................................ [ 25%]
................................................................................................................................................................ [ 30%]
................................................................................................................................................................ [ 35%]
.........................................................................................................................s.sssss.........................ss..... [ 40%]
................................................................................................................................................................ [ 45%]
................................................................................................................................................................ [ 51%]
.......................................................s........ssss..................................................................................Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 86 in test_set_deterministic_seeds_lightgbm
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 94 in test_set_deterministic_seeds_xgboost
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 103 in test_set_deterministic_seeds_catboost
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py", line 1833 in _inner
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
..Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 123 in test_set_deterministic_seeds_pytorch
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\unittest\mock.py", line 1833 in _inner
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
... [ 56%]
..........Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 420 in test_deterministic_numpy_operations
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
Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 426 in test_deterministic_numpy_operations
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 435 in test_deterministic_pandas_operations
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
Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 450 in test_deterministic_pandas_operations
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
...Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 514 in test_full_reproducibility_workflow
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 555 in test_reproducibility_with_different_seeds
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
Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 559 in test_reproducibility_with_different_seeds
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
.Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 568 in test_reproducibility_with_same_seed
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
Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_reproducibility.py", line 571 in test_reproducibility_with_same_seed
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
...............................Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 1035 in initialize_experiment_tracking
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 1124 in main
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_train.py", line 350 in test_main_success
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
......Windows fatal exception: access violation

Thread 0x00002a78 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\ml\model_monitor.py", line 227 in _monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003bbc (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003428 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\concurrent\futures\thread.py", line 81 in _worker
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002ca0 (most recent call first):
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 324 in wait
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 607 in wait
  File "C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\tqdm\_monitor.py", line 60 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00002770 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x000023e4 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Thread 0x00003ac8 (most recent call first):
  File "C:\Users\TU\Desktop\new project\N1V1\core\memory_manager.py", line 110 in _memory_monitoring_loop
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 953 in run
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 1016 in _bootstrap_inner
  File "C:\Users\TU\AppData\Local\Programs\Python\Python310\lib\threading.py", line 973 in _bootstrap

Current thread 0x000006a0 (most recent call first):
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
  File "C:\Users\TU\Desktop\new project\N1V1\ml\train.py", line 134 in set_deterministic_seeds
  File "C:\Users\TU\Desktop\new project\N1V1\tests\ml\test_train.py", line 550 in test_set_deterministic_seeds_basic
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
........................................................................................................... [ 61%]
.......................................................................s........................................................................sssss........... [ 66%]
................................................................................................................................................................ [ 71%]
................................................................................................................................................................ [ 76%]
................................................................................................................................................................ [ 81%]
.................s..............................................................................................................................2f
Triggering emergency memory cleanup
................. [ 86%]
................................................................................................................................................................ [ 91%]
................................................................................................................................................................ [ 97%]
...........................................................................................                                                                      [100%]
========================================================================== warnings summary ===========================================================================
venv\lib\site-packages\starlette\formparsers.py:10
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\starlette\formparsers.py:10: PendingDeprecationWarning: Please use `import python_multipart` instead.
    import multipart

venv\lib\site-packages\pydantic\_internal\_fields.py:149
  C:\Users\TU\Desktop\new project\N1V1\venv\lib\site-packages\pydantic\_internal\_fields.py:149: UserWarning: Field "model_version" has conflict with protected namespace "model_".

  You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
3111 passed, 20 skipped, 2 warnings in 827.74s (0:13:47)
sys:1: RuntimeWarning: coroutine '_cleanup_endpoint_on_exit' was never awaited
Object allocated at (most recent call last):
  File "<unknown>", lineno 0