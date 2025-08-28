# N1V1 Medium-Term Upgrade Roadmap

This file tracks the medium-term upgrades to implement for the N1V1 project. Work will be done step-by-step with small commits and tests after each change.

- [x] Risk Management Modes
  - [x] Add fixed_percent sizing mode
  - [x] Add volatility_based sizing mode (ATR/stddev)
  - [x] Add kelly_criterion sizing mode (simplified)
  - [x] Keep legacy fixed_fractional as default for backward compatibility
  - [ ] Add config.json entries and .env support for new risk params
- [x] Advanced Order Logic
  - [x] Trailing Stop Loss support in OrderManager
  - [x] Dynamic Take Profit (trend-strength based)
  - [x] Profit-based Re-entry logic
  - [ ] Toggleable via config
- [x] Portfolio / Multi-Pair Mode
  - [x] BotEngine: multi-pair handling
  - [x] Capital allocation per pair (configurable)
  - [ ] Backtester + TradeLogger multi-pair support
- [ ] Plugin Strategy System
  - [ ] Strategy registry/loader (auto-discover strategies/)
  - [ ] Select active strategies via config
- [ ] Integration Tests
  - [ ] End-to-end integration test(s) for full pipeline (dummy dataset)
  - [ ] Validate CSV/JSON outputs produced by TradeLogger/backtester
- [ ] Error Handling & Reliability
  - [ ] Retry logic for network/API calls (3 retries, backoff)
  - [ ] Safe-mode flag to skip trades on runtime errors
  - [ ] Catch & isolate strategy errors (don't stop bot)
  - [ ] Improve logging of full tracebacks in critical modules

Notes:
- All changes will be incremental and backward-compatible where possible.
- Tests (pytest) will be run after each change.
