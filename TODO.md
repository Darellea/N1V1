# N1V1 Short-Term Upgrade Roadmap

This checklist covers the short-term improvements to apply to the N1V1 project, executed step-by-step with commits after each major goal.

- [x] Analyze requirements
- [x] PEP8 & Linting
  - [x] Add `ruff` for linting
  - [x] Add `black` for formatting
  - [x] Add configuration files for ruff/black (if needed)
  - [x] Run linter and auto-format code
- [ ] Type Hints & Docstrings
  - [x] Add typing to core modules: `core/bot_engine.py`, `core/order_manager.py`, `strategies/*`, `backtest/backtester.py`, `risk/risk_manager.py`
  - [ ] Add clear docstrings for each core function (inputs/outputs, side-effects)
  - [ ] Run mypy (optional) or ruff type checks
- [ ] Config System
  - [ ] Add `.env` to store sensitive data (do not commit `.env`)
  - [ ] Use `python-dotenv` to load environment variables
  - [ ] Update `utils/config_loader.py` to support `.env` and env var overrides
- [ ] Trade Logger (CSV)
  - [ ] Update `utils/logger.py` to log trades with: timestamp, pair, action, size, entry price, exit price, PnL
  - [ ] Ensure logger works in live mode and backtester mode
  - [ ] Add rotating file handler for logs (optional)
- [ ] Backtester Metrics
  - [ ] Extend `backtest/backtester.py` to compute:
    - equity curve per trade
    - max drawdown
    - Sharpe ratio
    - profit factor
  - [ ] Export detailed results and metrics to `backtest/results.csv`
  - [ ] Add unit tests for new metrics
- [ ] Requirements files
  - [ ] Keep production dependencies in `requirements.txt`
  - [ ] Create `requirements-dev.txt` with `pytest`, `black`, `ruff`, `python-dotenv`
- [ ] Tests & CI
  - [ ] Run `pytest` after each major change
  - [ ] Fix issues found by tests or linters
- [ ] Final review & cleanup
  - [ ] Ensure backward compatibility where possible
  - [ ] Format codebase (black) and run ruff fix
  - [ ] Commit changes after each upgrade goal

Notes:
- Sensitive files: add `.env` to `.gitignore`.
- I will proceed goal-by-goal, updating files and running tests after each major change. I will wait for your confirmation before executing file edits or CLI commands affecting the environment.
