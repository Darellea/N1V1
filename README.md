# N1V1 Crypto Trading Framework

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code Style](https://img.shields.io/badge/code_style-black-black)

```mermaid
flowchart TD
    subgraph Ingress [Data Ingestion Layer]
        A[External Exchange APIs (Binance / CCXT / CSV)]
        B[Historical Data / Live Streams]
    end

    subgraph Processing [Core Engine Pipeline]
        C[Strategy Plugins / ML Signal Engines]
        D[Risk Management Layer]
        E[Order Router & Execution Logic]
    end

    subgraph Output [Execution Targets & Observability]
        F[Paper Trading / Backtest Engine]
        G[Live Exchange Execution]
        H[Logs / Metrics / Reporting]
    end

    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    F --> H
    G --> H
```

**Enterprise-grade automated cryptocurrency trading framework with institutional-level risk management, AI-powered optimization, and production-ready reliability.**

## Why This Exists

N1V1 addresses the critical need for sophisticated, scalable trading systems in the cryptocurrency markets. Traditional trading approaches lack the robustness required for 24/7 automated operation. This framework provides:

- **Institutional-grade risk controls** with circuit breaker systems and adaptive policies
- **AI-enhanced signal processing** using machine learning for market regime detection
- **Production reliability** with comprehensive monitoring, error recovery, and self-healing capabilities
- **Scalable architecture** supporting multiple strategies, exchanges, and concurrent operations
- **Enterprise monitoring** with Prometheus/Grafana integration and real-time dashboards

## Who Is This For?

| Persona                   | Profile                                                | How They Benefit                             |
| ------------------------- | ------------------------------------------------------ | -------------------------------------------- |
| Solo Quant Trader         | Builds and tests strategies alone                      | Fast iteration, automation, reproducibility  |
| Prop Firm / Desk Engineer | Scales execution for multiple strategies               | Modular, auditable, risk-managed             |
| ML / Research Engineer    | Trains and deploys predictive models into trading loop | Built-in signal hooks and inference pipeline |
| Plugin Developer          | Extends framework with new indicators or executors     | Well-defined extension contracts             |

## TL;DR — Key Features at a Glance

✅ Modular, Plug-and-Play Architecture
✅ Fully Async or Sync Execution Engine
✅ Strategy Plugins & ML Signal Hooks
✅ Backtest and Live Execution from Same Interface
✅ First-Class Risk Management & Circuit Breakers
✅ Extensible Configuration (YAML / ENV / CLI)
✅ Rich Logging, Observability & Debug Tracing

## Architecture Overview

```mermaid
graph TD
    A[Market Data Sources] --> B[Data Manager]
    B --> C[Signal Processor]
    C --> D[Risk Manager]
    D --> E[Order Manager]
    E --> F[Execution Engine]
    F --> G[Performance Tracker]
    G --> H[State Manager]
    H --> I[Dashboard/API]

    subgraph "Core Components"
        B
        C
        D
        E
        F
        G
        H
    end

    subgraph "Supporting Systems"
        J[ML Models] --> C
        K[Cache Layer] --> B
        L[Monitoring] --> H
        M[Alerting] --> D
    end
```

## Proven Results (Example Backtest / Live Run)

*Note: Results shown are from framework testing. Replace with your actual performance data.*

| Metric        | Value    |
| ------------- | -------- |
| Total Return  | +263.07% |
| Win Rate      | 69.44%   |
| Max Drawdown  | 18.47%   |
| Sharpe Ratio  | 8.68     |
| Profit Factor | 3.32     |

## System Dependencies

```mermaid
graph LR
    DataSource --> SignalEngine
    SignalEngine --> RiskManager
    RiskManager --> OrderExecutor
    OrderExecutor --> Exchange
    BacktestEngine --> ResultLogger
```

## Execution Lifecycle

```mermaid
sequenceDiagram
    participant U as User/Config
    participant BE as BotEngine
    participant DM as DataManager
    participant SP as SignalProcessor
    participant RM as RiskManager
    participant OM as OrderManager
    participant EX as Executor
    participant PT as PerformanceTracker

    U->>BE: Initialize with config
    BE->>DM: Fetch market data
    DM-->>BE: Return data
    BE->>SP: Generate signals
    SP-->>BE: Return signals
    BE->>RM: Evaluate risk
    RM-->>BE: Approve/Reject
    BE->>OM: Execute approved orders
    OM->>EX: Submit to exchange
    EX-->>OM: Execution result
    OM->>PT: Record performance
    PT-->>BE: Update metrics
    BE->>BE: Update state & repeat
```

## Repository Structure

```text
├── .coverage
├── .coveragerc
├── .env
├── .gitignore
├── ENHANCED_BINARY_MODEL_README.md
├── ENSEMBLE_README.md
├── INFO.txt
├── README
│   ├── README_CIRCUIT_BREAKER.md
│   ├── README_MONITORING_OBSERVABILITY.md
│   ├── README_REGIME_FORECASTING.md
│   ├── README_SELF_HEALING_ENGINE.md
│   ├── README_STRATEGY_GENERATOR.md
│   └── README_TESTING_FRAMEWORK.md
├── README.md
├── TODO.md
├── acceptance_reports
│   └── acceptance_summary.json
├── anomalies.json
├── api
│   ├── __init__.py
│   ├── app.py
│   ├── middleware.py
│   ├── models.py
│   └── schemas.py
├── backtest
│   ├── __init__.py
│   └── backtester.py
├── benchmark_data_optimizations.py
├── benchmark_results.json
├── chaos_reports.json
├── check_sizes.py
├── config.json
├── config_binary_integration_example.json
├── config_data_expansion.json
├── config_ensemble_example.json
├── config_retraining_scheduler.json
├── core
│   ├── __init__.py
│   ├── alert_rules_manager.py
│   ├── async_optimizer.py
│   ├── binary_model_integration.py
│   ├── binary_model_metrics.py
│   ├── bot_engine.py
│   ├── cache.py
│   ├── circuit_breaker.py
│   ├── component_factory.py
│   ├── config_manager.py
│   ├── contracts.py
│   ├── dashboard_manager.py
│   ├── data_expansion_manager.py
│   ├── data_manager.py
│   ├── data_processor.py
│   ├── diagnostics.py
│   ├── distributed_system.py
│   ├── ensemble_manager.py
│   ├── execution
│   │   ├── __init__.py
│   │   ├── adaptive_pricer.py
│   │   ├── backtest_executor.py
│   │   ├── base_executor.py
│   │   ├── dca_executor.py
│   │   ├── execution_types.py
│   │   ├── live_executor.py
│   │   ├── order_processor.py
│   │   ├── paper_executor.py
│   │   ├── retry_manager.py
│   │   ├── smart_layer.py
│   │   ├── smart_order_executor.py
│   │   ├── twap_executor.py
│   │   ├── validator.py
│   │   └── vwap_executor.py
│   ├── healthcheck.py
│   ├── interfaces.py
│   ├── logging_utils.py
│   ├── management
│   │   ├── __init__.py
│   │   ├── portfolio_manager.py
│   │   └── reliability_manager.py
│   ├── memory_manager.py
│   ├── metrics_collector.py
│   ├── metrics_endpoint.py
│   ├── model_monitor.py
│   ├── order_executor.py
│   ├── order_manager.py
│   ├── performance_monitor.py
│   ├── performance_profiler.py
│   ├── performance_reports.py
│   ├── performance_tracker.py
│   ├── self_healing_engine.py
│   ├── signal_processor.py
│   ├── signal_router
│   │   ├── __init__.py
│   │   ├── event_bus.py
│   │   ├── events.py
│   │   ├── retry_hooks.py
│   │   ├── route_policies.py
│   │   ├── router.py
│   │   └── signal_validators.py
│   ├── signal_router.py
│   ├── state_manager.py
│   ├── task_manager.py
│   ├── timeframe_manager.py
│   ├── trading_coordinator.py
│   ├── types
│   │   ├── __init__.py
│   │   └── order_types.py
│   ├── types.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── config_utils.py
│   │   └── error_utils.py
│   └── watchdog.py
├── data
│   ├── __init__.py
│   ├── constants.py
│   ├── data_fetcher.py
│   ├── dataset_versioning.py
│   ├── historical_loader.py
│   ├── interfaces.py
│   └── performance_baselines.json
├── debug_regression.py
├── demo
│   ├── demo_circuit_breaker.py
│   ├── demo_cross_pair_validation.py
│   ├── demo_monitoring_integration.py
│   ├── demo_performance_optimization.py
│   ├── demo_regime_forecasting.py
│   ├── demo_self_healing_engine.py
│   ├── demo_strategy_generator.py
│   ├── demo_time_utils.py
│   ├── test_anomaly_integration.py
│   ├── test_execution_demo.py
│   ├── test_metrics_demo.py
│   └── test_simple_metrics.py
├── demo_calibrated_model_config.json
├── demo_docstring_standardization.py
├── demo_duplication_elimination.py
├── demo_training_results.json
├── deploy
│   ├── Dockerfile.api
│   ├── Dockerfile.core
│   ├── Dockerfile.ml
│   ├── Dockerfile.ml-serving
│   ├── Dockerfile.monitoring
│   ├── canary.sh
│   ├── docker-compose.dev.yml
│   └── k8s
│       ├── configmap.yaml
│       ├── deployment-api.yaml
│       ├── deployment-core.yaml
│       ├── hpa.yaml
│       ├── ingress.yaml
│       ├── ml-serving.yaml
│       ├── namespace.yaml
│       ├── services.yaml
│       └── statefulset-ml.yaml
├── error1.txt
├── etc
│   └── passwd
├── feature_importance.png
├── horizon_threshold_test.txt
├── knowledge_base
│   ├── __init__.py
│   ├── adaptive.py
│   ├── manager.py
│   ├── schema.py
│   └── storage.py
├── loc_analysis.json
├── main.py
├── ml
│   ├── __init__.py
│   ├── features.py
│   ├── indicators.py
│   ├── ml_filter.py
│   ├── model_loader.py
│   ├── model_monitor.py
│   ├── serving.py
│   ├── train.py
│   └── trainer.py
├── notifier
│   ├── __init__.py
│   ├── discord_bot.py
│   └── test_discord_send.py
├── optimization
│   ├── __init__.py
│   ├── asset_selector.py
│   ├── base_optimizer.py
│   ├── bayesian_optimizer.py
│   ├── config.py
│   ├── cross_asset_validation.py
│   ├── cross_asset_validator.py
│   ├── distributed_evaluator.py
│   ├── genetic_optimizer.py
│   ├── genome.py
│   ├── market_data_fetcher.py
│   ├── optimizer_factory.py
│   ├── rl_optimizer.py
│   ├── strategy_factory.py
│   ├── strategy_generator.py
│   ├── validation_criteria.py
│   ├── validation_results.py
│   └── walk_forward.py
├── portfolio
│   ├── __init__.py
│   ├── allocation_engine.py
│   ├── allocator.py
│   ├── hedging.py
│   ├── performance_aggregator.py
│   ├── portfolio_manager.py
│   └── strategy_ensemble.py
├── predictive_models
│   ├── __init__.py
│   ├── predictive_model_manager.py
│   ├── price_predictor.py
│   ├── types.py
│   ├── volatility_predictor.py
│   └── volume_predictor.py
├── pyproject.toml
├── pytest.ini
├── reporting
│   ├── __init__.py
│   ├── metrics.py
│   ├── scheduler.py
│   └── sync.py
├── requirements-dev.txt
├── requirements.txt
├── risk
│   ├── __init__.py
│   ├── adaptive_policy.py
│   ├── anomaly_detector.py
│   ├── risk_manager.py
│   └── utils.py
├── scheduler
│   ├── diagnostic_scheduler.py
│   └── retraining_scheduler.py
├── scripts
│   ├── __init__.py
│   ├── chaos_test.sh
│   ├── count_loc.py
│   ├── demo_binary_integration.py
│   ├── demo_binary_monitoring.py
│   ├── demo_binary_training.py
│   ├── demo_calibration_threshold.py
│   ├── migrate_to_binary_labels.py
│   ├── run_acceptance_tests.py
│   ├── run_data_expansion.py
│   ├── run_final_audit.py
│   ├── run_model_benchmarks.py
│   ├── run_profiling.py
│   ├── run_retraining_scheduler.py
│   ├── test_binary_labels.py
│   └── tree.py
├── shap_feature_importance.txt
├── start.bat
├── strategies
│   ├── __init__.py
│   ├── atr_breakout_strategy.py
│   ├── base_strategy.py
│   ├── bollinger_reversion_strategy.py
│   ├── donchian_breakout_strategy.py
│   ├── ema_cross_strategy.py
│   ├── generated
│   │   └── __init__.py
│   ├── indicators_cache.py
│   ├── keltner_channel_strategy.py
│   ├── macd_strategy.py
│   ├── mixins.py
│   ├── obv_strategy.py
│   ├── regime
│   │   ├── market_regime.py
│   │   ├── regime_forecaster.py
│   │   └── strategy_selector.py
│   ├── rsi_strategy.py
│   ├── stochastic_strategy.py
│   └── vwap_pullback_strategy.py
├── test_benchmark_results.json
├── test_data.csv
├── test_logging_demo.py
├── test_model.model_card.json
├── test_model_config.json
├── test_model_new.model_card.json
├── test_model_new_config.json
├── test_output.log.1
├── test_results.json
├── test_system_validation.py
├── test_validation_data.csv
├── tests
│   ├── acceptance
│   │   ├── test_docs.py
│   │   ├── test_ml_quality.py
│   │   ├── test_scalability.py
│   │   ├── test_slo.py
│   │   └── test_stability.py
│   ├── api
│   │   ├── test_api_app.py
│   │   └── test_endpoints.py
│   ├── backtest
│   │   ├── test_backtest_executor.py
│   │   ├── test_backtester.py
│   │   └── test_regime_aware_backtester.py
│   ├── conftest.py
│   ├── core
│   │   ├── test_alert_rules_manager.py
│   │   ├── test_alerting.py
│   │   ├── test_algorithmic_correctness.py
│   │   ├── test_binary_integration_enhanced.py
│   │   ├── test_binary_model_metrics.py
│   │   ├── test_bot_engine_comprehensive.py
│   │   ├── test_cache_comprehensive.py
│   │   ├── test_circuit_breaker.py
│   │   ├── test_config_manager.py
│   │   ├── test_dashboard_manager.py
│   │   ├── test_dependency_injection.py
│   │   ├── test_diagnostics.py
│   │   ├── test_ensemble_manager.py
│   │   ├── test_event_driven_architecture.py
│   │   ├── test_execution.py
│   │   ├── test_journal_writer.py
│   │   ├── test_live_executor.py
│   │   ├── test_logging_and_resources.py
│   │   ├── test_memory_leak_stress.py
│   │   ├── test_monitoring_observability.py
│   │   ├── test_order_manager.py
│   │   ├── test_order_processor.py
│   │   ├── test_paper_executor.py
│   │   ├── test_performance_optimization.py
│   │   ├── test_regression.py
│   │   ├── test_reliability_manager.py
│   │   ├── test_safe_mode.py
│   │   ├── test_self_healing_engine.py
│   │   ├── test_signal_router.py
│   │   ├── test_signal_router_facade.py
│   │   ├── test_task_manager.py
│   │   ├── test_timeframe_manager.py
│   │   ├── test_trading_signal_amount.py
│   └── test_types.py
│   ├── data
│   │   ├── test_data.py
│   │   ├── test_data_fetcher.py
│   │   ├── test_data_fixes.py
│   │   ├── test_data_module_refactoring.py
│   │   ├── test_data_security.py
│   │   └── test_historical_loader.py
│   ├── edge_case_testing.py
│   ├── execution
│   │   ├── test_smart_layer.py
│   │   └── test_validator.py
│   ├── integration
│   │   ├── test_cross_feature_integration.py
│   │   ├── test_distributed_system.py
│   │   ├── test_ml_serving_integration.py
│   │   └── test_order_flow_integration.py
│   ├── integration_test_framework.py
│   ├── knowledge_base
│   │   └── test_knowledge_base.py
│   ├── ml
│   │   ├── test_features.py
│   │   ├── test_indicators.py
│   │   ├── test_ml.py
│   │   ├── test_ml_artifact_model_card.py
│   │   ├── test_ml_filter.py
│   │   ├── test_ml_signal_router.py
│   │   ├── test_model_loader.py
│   │   ├── test_model_monitor.py
│   │   ├── test_predictive_models.py
│   │   ├── test_reproducibility.py
│   │   ├── test_serving.py
│   │   ├── test_train.py
│   │   └── test_trainer.py
│   ├── notifier
│   │   ├── test_discord_integration.py
│   │   └── test_discord_notifier.py
│   ├── optimization
│   │   ├── test_asset_selector.py
│   │   ├── test_async_optimizer.py
│   │   ├── test_cross_asset_validation.py
│   │   ├── test_optimization.py
│   │   └── test_walk_forward.py
│   ├── portfolio
│   │   ├── test_allocation_engine.py
│   │   ├── test_portfolio.py
│   │   ├── test_strategy_ensemble.py
│   ├── reporting
│   ├── risk
│   │   ├── test_adaptive_policy.py
│   │   ├── test_adaptive_risk.py
│   │   ├── test_anomaly_detection.py
│   │   ├── test_anomaly_detector.py
│   │   ├── test_risk.py
│   │   └── test_risk_manager_integration.py
│   ├── run_comprehensive_tests.py
│   ├── run_smoke_tests.py
│   ├── scheduler
│   ├── security
│   │   ├── test_core_security.py
│   │   ├── test_key_management.py
│   │   ├── test_order_invariants.py
│   │   ├── test_secret_manager.py
│   │   └── test_secure_logging.py
│   ├── strategies
│   │   ├── test_market_regime.py
│   │   ├── test_regime_forecaster.py
│   │   ├── test_strategies.py
│   │   ├── test_strategy.py
│   │   ├── test_strategy_generator.py
│   │   ├── test_strategy_integration.py
│   │   └── test_strategy_selector.py
│   ├── stress
│   │   ├── chaos_tests.py
│   │   ├── test_cluster_scaling.py
│   │   └── test_load.py
│   ├── test_integration.py
│   ├── test_main.py
│   └── utils
│       ├── test_adapter.py
│       ├── test_circular_import_fix.py
│       ├── test_config_loader.py
│       ├── test_demo_time_utils.py
│       ├── test_docstring_standardization.py
│       ├── test_logger.py
│       ├── test_retry.py
│       ├── test_time.py
│       └── test_time_utils.py
├── tools
│   └── check_imports.py
├── training_results.json
├── training_results_backtest.json
├── training_results_new.json
├── training_results_shap12.json
├── training_results_stress.json
├── training_results_unbalance.json
├── training_results_weights.json
├── tree.txt
├── utils
│   ├── __init__.py
│   ├── adapter.py
│   ├── code_quality.py
│   ├── config_factory.py
│   ├── config_generator.py
│   ├── config_loader.py
│   ├── constants.py
│   ├── dependency_manager.py
│   ├── docstring_standardizer.py
│   ├── duplication_analyzer.py
│   ├── error_handler.py
│   ├── error_handling_utils.py
│   ├── final_auditor.py
│   ├── logger.py
│   ├── logging_manager.py
│   ├── logging_utils.py
│   ├── retry.py
│   ├── security.py
│   └── time.py
```

## Module Breakdown

### Core Framework (`core/`)
**Purpose**: Core framework backbone with async processing, event-driven architecture, risk management, monitoring, and trading coordination.

**Public Entry Points**: `get_component_factory()`, `get_config_manager()`, `get_memory_manager()`

**Key Classes / Functions**:
- `BotEngine`: Main trading engine with async processing and event-driven architecture
- `CircuitBreaker`: Advanced circuit breaker system for automatic trading suspension
- `PerformanceMonitor`: System performance monitoring with metrics collection
- `TradingCoordinator`: Coordinates multiple trading strategies and execution
- `SignalProcessor`: Processes and routes trading signals across strategies
- `OrderManager`: Manages order lifecycle from creation to execution
- `StateManager`: Maintains system state and configuration persistence
- `TaskManager`: Asynchronous task scheduling and execution
- `MemoryManager`: Memory optimization and leak prevention
- `Cache`: High-performance caching system for market data
- `ConfigManager`: Configuration management with validation
- `DashboardManager`: Web dashboard management and real-time updates
- `Diagnostics`: System diagnostics and health monitoring
- `SelfHealingEngine`: Automatic error recovery and system stabilization
- `TimeframeManager`: Multi-timeframe data management and synchronization
- `Watchdog`: System watchdog for process monitoring and restart

**Processing Flow**: Market data fetch → Signal generation → Risk evaluation → Order execution → State update

**Dependencies**: pandas, numpy, asyncio, prometheus_client

**Extension / Plug Points**: Interfaces for data_manager, signal_processor, risk_manager, order_executor

### Data Management (`data/`)
**Purpose**: Data acquisition, processing, and management for historical and real-time market data.

**Public Entry Points**: `DataFetcher`, `HistoricalDataLoader`, `DatasetVersionManager`

**Key Classes / Functions**:
- `DataFetcher`: Fetches data from exchanges with rate limiting
- `HistoricalDataLoader`: Loads historical data with pagination
- `DatasetVersionManager`: Versions datasets with integrity checking

**Processing Flow**: Fetch data from exchange → Validate and parse → Cache data → Return DataFrame

**Dependencies**: ccxt, pandas, aiofiles

**Extension / Plug Points**: `IDataFetcher` interface for custom data sources

### API & Web Interface (`api/`)
**Purpose**: REST API for web interface, monitoring, and external integration.

**Public Entry Points**: `app.py` (FastAPI app), `health_check`, `metrics`, `dashboard`

**Key Classes / Functions**:
- FastAPI app with endpoints for status, orders, signals, equity, performance

**Processing Flow**: HTTP request → Authentication → Rate limiting → Handler → Response

**Dependencies**: fastapi, uvicorn, sqlalchemy

**Extension / Plug Points**: Middleware for custom authentication, rate limiting

### Backtesting (`backtest/`)
**Purpose**: Backtesting framework for strategy evaluation with historical data.

**Public Entry Points**: `Backtester` class, `compute_backtest_metrics`, export functions

**Key Classes / Functions**:
- `Backtester`: Runs backtests on strategies
- `compute_backtest_metrics`: Calculates Sharpe, drawdown, etc.
- Export functions for equity progression and results

**Processing Flow**: Strategy genome + market data → Simulate trades → Calculate metrics → Export results

**Dependencies**: pandas, numpy

**Extension / Plug Points**: Strategy genome interface for custom strategies

### Trading Strategies (`strategies/`)
**Purpose**: Trading strategy implementations using technical indicators.

**Public Entry Points**: `BaseStrategy` class, specific strategy classes like `EMACrossStrategy`, `RSIStrategy`, etc.

**Key Classes / Functions**:
- `BaseStrategy`: Abstract base class defining strategy interface
- `EMACrossStrategy`: Exponential moving average crossover strategy
- `RSIStrategy`: Relative strength index mean-reversion strategy
- `MACDStrategy`: Moving average convergence divergence momentum strategy
- `BollingerReversionStrategy`: Bollinger bands mean-reversion strategy
- `StochasticStrategy`: Stochastic oscillator momentum strategy
- `KeltnerChannelStrategy`: Keltner channel volatility breakout strategy
- `DonchianBreakoutStrategy`: Donchian channel breakout strategy
- `ATRBreakoutStrategy`: Average true range volatility breakout strategy
- `VWAPPullbackStrategy`: Volume weighted average price pullback strategy
- `OBVStrategy`: On-balance volume momentum strategy
- `IndicatorsCache`: Technical indicator caching and optimization

**Processing Flow**: Market data → Calculate indicators → Generate signals → Return TradingSignal list

**Dependencies**: pandas, numpy, ta (technical analysis)

**Extension / Plug Points**: `BaseStrategy` inheritance for custom strategies

### Utilities (`utils/`)
**Purpose**: Utility functions for configuration, logging, error handling, security, and code quality.

**Public Entry Points**: Various utility functions like `get_config_factory()`, `get_error_handler()`, `setup_logging()`, etc.

**Key Classes / Functions**:
- `ConfigFactory`: Configuration file generation and management
- `ErrorHandler`: Centralized error handling and logging
- `LoggingManager`: Advanced logging with rotation and filtering
- `Security`: Security utilities and encryption
- `CodeQualityAnalyzer`: Code complexity and quality analysis
- `DocstringStandardizer`: Documentation standardization
- `DuplicationAnalyzer`: Code duplication analysis and reporting
- `DependencyManager`: Dependency injection and management

**Processing Flow**: Varies by utility, e.g., config loading → validation → caching

**Dependencies**: ast, logging, pydantic

**Extension / Plug Points**: Plugin interfaces for custom loggers, error handlers, etc.

### Machine Learning (`ml/`)
**Purpose**: Machine learning components for feature engineering, model training, and signal filtering.

**Public Entry Points**: `FeatureExtractor`, `MLFilter`, `train_model_binary`, etc.

**Key Classes / Functions**:
- `FeatureExtractor`: Feature engineering from market data
- `MLFilter`: Signal validation using ML models
- `ModelMonitor`: Performance tracking for ML models
- Training pipeline functions for model development

**Processing Flow**: Data → Feature extraction → Model training/prediction → Signal filtering

**Dependencies**: sklearn, xgboost, pandas, numpy

**Extension / Plug Points**: `MLModel` abstract class for custom models

### Risk Management (`risk/`)
**Purpose**: Risk management with position sizing, anomaly detection, and adaptive policies.

**Public Entry Points**: `RiskManager`, `AnomalyDetector`, `AdaptiveRiskPolicy`

**Key Classes / Functions**:
- `RiskManager`: Position sizing and limits enforcement
- `AnomalyDetector`: Market anomaly detection using statistics
- `AdaptiveRiskPolicy`: Dynamic risk adjustment based on conditions

**Processing Flow**: Signal → Risk evaluation → Position sizing → Anomaly check → Approval/Rejection

**Dependencies**: pandas, numpy, scipy

**Extension / Plug Points**: `BaseAnomalyDetector` for custom detectors

### Portfolio Management (`portfolio/`)
**Purpose**: Portfolio management with allocation, rebalancing, and hedging.

**Public Entry Points**: `PortfolioManager`, `AllocationEngine`, `StrategyEnsembleManager`

**Key Classes / Functions**:
- `PortfolioManager`: Position tracking and rebalancing
- `AllocationEngine`: Asset allocation algorithms
- `PortfolioHedger`: Risk hedging strategies
- `PerformanceAggregator`: Portfolio performance aggregation

**Processing Flow**: Positions → Performance calculation → Rebalancing triggers → Allocation adjustment → Trade execution

**Dependencies**: pandas, numpy, scipy

**Extension / Plug Points**: `CapitalAllocator` abstract class for custom allocators

### Optimization (`optimization/`)
**Purpose**: Strategy optimization using genetic algorithms, Bayesian optimization, walk-forward analysis, and cross-asset validation.

**Public Entry Points**: `OptimizerFactory`, `create_walk_forward_optimizer`, etc.

**Key Classes / Functions**:
- `BaseOptimizer`: Abstract base class for optimization algorithms
- `GeneticOptimizer`: Genetic algorithm-based parameter optimization
- `BayesianOptimizer`: Bayesian optimization for hyperparameter tuning
- `WalkForwardOptimizer`: Walk-forward analysis for strategy validation
- `CrossAssetValidator`: Cross-asset validation and overfitting prevention

**Processing Flow**: Strategy genome → Parameter optimization → Backtest evaluation → Fitness scoring → Next generation

**Dependencies**: deap, scikit-optimize, pandas, numpy

**Extension / Plug Points**: `BaseOptimizer` for custom optimizers

## Configuration & Extensibility

### Core Configuration
```json
{
  "exchange": {
    "name": "binance",
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "testnet": true
  },
  "risk_management": {
    "max_position_size": 0.02,
    "max_drawdown": 0.1,
    "circuit_breaker_enabled": true,
    "circuit_breaker_threshold": 0.05
  },
  "monitoring": {
    "prometheus_enabled": true,
    "grafana_enabled": true,
    "alerting_enabled": true
  },
  "strategies": {
    "active_strategies": ["ema_cross", "rsi", "macd"],
    "max_concurrent": 5
  }
}
```

### Extension Points
- **Strategies**: Inherit from `BaseStrategy` for custom trading logic
- **Risk Managers**: Implement `RiskManagerInterface` for custom risk controls
- **Data Sources**: Implement `IDataFetcher` for custom market data providers
- **Optimizers**: Extend `BaseOptimizer` for custom optimization algorithms
- **ML Models**: Extend `MLModel` for custom machine learning algorithms
- **Allocators**: Implement `CapitalAllocator` for custom portfolio allocation

## Security & Safety Model

This framework is designed with safety-first execution:

* **Dry-Run Mode & Sandbox Execution**
* **Circuit Breakers prevent runaway orders**
* **Config Locking to avoid accidental misfires**
* **Explicit Exchange Credential Scopes**

## Example: Minimal Custom Strategy

```python
from strategies import BaseStrategy

class MyBreakoutStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 20)

    async def generate_signals(self, market_data):
        signals = []
        for symbol, data in market_data.items():
            if len(data) < self.lookback_period:
                continue

            current_price = data.iloc[-1]['close']
            high_20 = data.iloc[-self.lookback_period:]['high'].max()
            low_20 = data.iloc[-self.lookback_period:]['low'].min()

            if current_price > high_20:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'strength': 1.0,
                    'reason': f'Breakout above {self.lookback_period}-period high'
                })
            elif current_price < low_20:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'strength': 1.0,
                    'reason': f'Breakout below {self.lookback_period}-period low'
                })

        return signals
```

## System Requirements

| Component      | Requirement                                      |
| -------------- | ------------------------------------------------ |
| Python Version | 3.10+                                            |
| Supported OS   | Linux, macOS, Windows                            |
| Dependencies   | Listed in `requirements.txt` or `pyproject.toml` |
| CPU vs GPU     | CPU only (unless strategy uses ML with GPU)      |

## Usage / Quick Start

### Installation
```bash
git clone https://github.com/Darellea/N1V1.git
cd N1V1
pip install -r requirements.txt
```

### Basic Usage
```bash
# Start trading engine
python main.py

# Run backtesting
python main.py --mode backtest --strategy ema_cross

# Start with web interface
python main.py --api
```

### Configuration
```bash
cp config.json.example config.json
# Edit config.json with your settings
```

## Example Flow / How It Works in Practice

1. **Initialization**: Framework loads configuration and initializes components
2. **Data Acquisition**: DataManager fetches market data from exchanges
3. **Signal Generation**: Strategies analyze data and generate trading signals
4. **Risk Assessment**: RiskManager evaluates signals against risk parameters
5. **Order Execution**: Approved signals are converted to orders and executed
6. **Performance Tracking**: Results are recorded and performance metrics updated
7. **State Management**: System state is persisted and dashboards updated
8. **Monitoring**: Continuous monitoring detects anomalies and triggers alerts
9. **Optimization**: Background processes optimize strategies and parameters

## Troubleshooting & Common Pitfalls

### Circuit Breaker Triggering Frequently
```python
# Adjust sensitivity in config
{
  "circuit_breaker": {
    "equity_drawdown_threshold": 0.05,
    "consecutive_losses_threshold": 5,
    "cooling_period_minutes": 10
  }
}
```

### High Memory Usage
- Reduce cache sizes in configuration
- Enable memory monitoring
- Process data in smaller batches

### Slow Performance
- Enable vectorization optimizations
- Increase concurrent processing limits
- Profile with built-in profiler

### Exchange Connection Issues
- Verify API credentials
- Check rate limits
- Enable retry mechanisms

## Glossary of Terms

- **Signal**: Trading recommendation generated by a strategy
- **Position**: Open trade with unrealized P&L
- **Drawdown**: Peak-to-trough decline in equity
- **Sharpe Ratio**: Risk-adjusted return measure
- **Circuit Breaker**: Automatic trading suspension system
- **Walk-Forward**: Out-of-sample testing methodology
- **Regime Detection**: Market condition classification
- **Ensemble**: Combination of multiple strategies/models

## License / Contribution Notes

**License**: MIT License

**Contributing**:
- Fork repository
- Create feature branch
- Add comprehensive tests (95%+ coverage)
- Update documentation
- Submit pull request

**Code Standards**:
- Black formatting
- Flake8 linting
- MyPy type checking
- Comprehensive documentation

## Roadmap / Planned Enhancements

* [ ] Strategy Marketplace / Plugin Registry
* [ ] Web Dashboard for Live Monitoring
* [ ] Auto-ML Strategy Optimizer
* [ ] Distributed / Multi-Process Backtesting Engine
* [ ] Exchange Simulator with Slippage & Latency Modeling

---

**Framework Version: 1.0.0 | Built for quantitative traders and algorithmic funds**
