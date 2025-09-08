# Ensemble Strategies Implementation

This document describes the implementation of Ensemble Strategies in the Spot Trading Framework.

## Overview

The Ensemble Manager allows combining multiple trading strategies to make more robust trading decisions. It supports different voting mechanisms and dynamic weight updates based on strategy performance.

## Features

### Voting Mechanisms

1. **Majority Vote**: Simple majority consensus (e.g., 2/3 strategies must agree)
2. **Weighted Vote**: Performance-based weights determine influence
3. **Confidence Average**: ML/indicator confidence scores averaged with threshold

### Dynamic Weight Updates

- Weights can be static (from config) or dynamic (based on performance)
- Performance metrics: Sharpe ratio, win rate, profit factor
- Automatic weight recalculation after backtesting/live trading

### Integration

- Seamlessly integrates with existing SignalRouter
- Works with all existing strategies
- Enhanced logging for ensemble decisions
- Backtesting comparison between individual and ensemble performance

## Configuration

### Basic Configuration

```json
{
  "ensemble": {
    "enabled": true,
    "mode": "weighted_vote",
    "dynamic_weights": true,
    "strategies": [
      {
        "id": "rsi_strategy",
        "weight": 0.4
      },
      {
        "id": "ema_cross_strategy",
        "weight": 0.3
      },
      {
        "id": "ml_filter",
        "weight": 0.3
      }
    ],
    "thresholds": {
      "confidence": 0.6,
      "vote_ratio": 0.66
    }
  }
}
```

### Configuration Parameters

- `enabled`: Enable/disable ensemble processing
- `mode`: Voting mechanism (`majority_vote`, `weighted_vote`, `confidence_average`)
- `dynamic_weights`: Enable automatic weight updates based on performance
- `strategies`: List of strategy IDs and their initial weights
- `thresholds`:
  - `confidence`: Minimum confidence score for confidence averaging
  - `vote_ratio`: Minimum vote ratio for majority/weighted voting

## Architecture

### Core Components

1. **EnsembleManager** (`core/ensemble_manager.py`)
   - Main ensemble logic and voting mechanisms
   - Strategy registration and weight management
   - Performance-based weight updates

2. **SignalRouter Integration** (`core/signal_router.py`)
   - Checks ensemble manager before processing individual signals
   - Passes ensemble signals through normal validation pipeline

3. **Backtester Integration** (`backtest/backtester.py`)
   - `compare_ensemble_vs_individual()` function for performance comparison
   - Enhanced metrics calculation for ensemble vs individual strategies

### Key Classes

#### EnsembleManager

```python
from core.ensemble_manager import EnsembleManager, VotingMode

# Initialize
config = {"enabled": True, "mode": "weighted_vote", ...}
ensemble_manager = EnsembleManager(config)

# Register strategies
ensemble_manager.register_strategy("rsi_strategy", rsi_strategy_instance)

# Get ensemble signal
ensemble_signal = ensemble_manager.get_ensemble_signal(market_data)

# Update weights based on performance
performance_metrics = {
    "rsi_strategy": {"sharpe_ratio": 2.0, "win_rate": 0.8}
}
ensemble_manager.update_weights(performance_metrics)
```

#### Voting Modes

- **MAJORITY_VOTE**: Simple majority consensus
- **WEIGHTED_VOTE**: Performance-weighted voting
- **CONFIDENCE_AVERAGE**: Confidence score averaging

## Usage Examples

### Basic Usage

```python
from core.ensemble_manager import EnsembleManager

# Create ensemble manager
ensemble_config = {
    "enabled": True,
    "mode": "weighted_vote",
    "strategies": [
        {"id": "strategy1", "weight": 0.5},
        {"id": "strategy2", "weight": 0.3},
        {"id": "strategy3", "weight": 0.2}
    ]
}
ensemble_manager = EnsembleManager(ensemble_config)

# Register strategies
ensemble_manager.register_strategy("strategy1", strategy1_instance)
ensemble_manager.register_strategy("strategy2", strategy2_instance)
ensemble_manager.register_strategy("strategy3", strategy3_instance)

# Generate ensemble signal
market_data = {"ohlcv": df, "features": feature_df}
ensemble_signal = ensemble_manager.get_ensemble_signal(market_data)
```

### Backtesting Comparison

```python
from backtest.backtester import compare_ensemble_vs_individual

# Run backtests for individual strategies
individual_results = {
    "strategy1": run_backtest(strategy1),
    "strategy2": run_backtest(strategy2),
    "ensemble": run_backtest(ensemble_strategy)
}

# Compare performance
comparison = compare_ensemble_vs_individual(
    ensemble_results["ensemble"],
    {k: v for k, v in individual_results.items() if k != "ensemble"}
)

print(f"Ensemble improvement: {comparison['comparison']['return_improvement']}")
```

## Signal Processing Flow

1. **Individual Strategy Signals**: Each registered strategy generates signals
2. **Ensemble Decision**: EnsembleManager combines signals using voting mechanism
3. **Signal Validation**: Ensemble signal passes through normal SignalRouter validation
4. **Risk Management**: Standard risk checks applied
5. **Order Execution**: Signal processed normally if it passes all checks

## Logging

Ensemble decisions are logged with detailed information:

```json
{
  "decision": "buy",
  "contributing_strategies": ["rsi_strategy", "ml_filter"],
  "vote_counts": {"buy": 2, "sell": 0, "hold": 0},
  "total_weight": 2.0,
  "confidence_score": 0.8,
  "voting_mode": "weighted_vote",
  "strategy_weights": {"rsi_strategy": 0.6, "ml_filter": 0.4}
}
```

## Performance Metrics

The ensemble system tracks:

- **Individual Strategy Metrics**: Sharpe ratio, win rate, profit factor
- **Ensemble Metrics**: Combined performance across all strategies
- **Weight Adjustments**: Automatic weight updates based on performance
- **Comparison Metrics**: Ensemble vs individual strategy performance

## Testing

Comprehensive test suite included:

```bash
# Run all ensemble tests
python -m pytest tests/test_ensemble_manager.py -v

# Run integration tests
python -m pytest tests/test_ensemble_manager.py::TestEnsembleIntegration -v
```

### Test Coverage

- Voting mechanism accuracy
- Weight calculation and updates
- Signal processing edge cases
- Integration with existing strategies
- Performance metric calculations
- Error handling and recovery

## Benefits

1. **Improved Decision Quality**: Combines multiple strategies for better signals
2. **Risk Reduction**: Consensus-based decisions reduce individual strategy bias
3. **Adaptive Learning**: Dynamic weights adjust to strategy performance
4. **Backtesting Insights**: Compare ensemble vs individual performance
5. **Seamless Integration**: Works with existing trading framework

## Future Enhancements

- Machine learning-based weight optimization
- Strategy correlation analysis
- Real-time performance monitoring
- Advanced voting mechanisms (e.g., Bayesian voting)
- Strategy addition/removal based on performance thresholds

## Files Modified/Created

- `core/ensemble_manager.py` - Main ensemble logic
- `core/signal_router.py` - Integration with signal processing
- `utils/config_loader.py` - Configuration support
- `backtest/backtester.py` - Performance comparison utilities
- `tests/test_ensemble_manager.py` - Comprehensive test suite
- `config_ensemble_example.json` - Example configuration

## Predictive Models Integration

The framework now includes advanced predictive models that can filter trading signals based on market predictions. These models use machine learning and statistical techniques to predict price direction, volatility, and volume surges.

### Features

#### Price Direction Classification
- **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Prediction**: Next candle direction (up/down/neutral)
- **Features**: Technical indicators, price momentum, volume patterns

#### Volatility Forecasting
- **Models**: GARCH (statistical), ML regression (Random Forest, XGBoost, LightGBM)
- **Prediction**: Next N-candle volatility regime (high/low)
- **Features**: Historical volatility, ATR, Bollinger Band width

#### Volume Surge Detection
- **Models**: Z-score threshold, ML classifiers
- **Prediction**: Volume surge detection (true/false)
- **Features**: Volume ratios, z-scores, price-volume correlation

### Configuration

```json
{
  "predictive_models": {
    "enabled": true,
    "confidence_threshold": 0.5,
    "models": {
      "price_direction": {
        "enabled": true,
        "type": "lightgbm",
        "confidence_threshold": 0.6,
        "lookback": 50,
        "model_path": "models/price_lightgbm.pkl",
        "scaler_path": "models/price_scaler.pkl"
      },
      "volatility": {
        "enabled": true,
        "type": "garch",
        "forecast_horizon": 5,
        "threshold": 0.02,
        "confidence_threshold": 0.6,
        "lookback": 100,
        "model_path": "models/volatility_garch.pkl",
        "scaler_path": "models/volatility_scaler.pkl",
        "block_high_volatility": false
      },
      "volume_surge": {
        "enabled": true,
        "type": "zscore",
        "threshold": 2.5,
        "confidence_threshold": 0.6,
        "lookback": 50,
        "model_path": "models/volume_zscore.pkl",
        "scaler_path": "models/volume_scaler.pkl",
        "require_surge": false
      }
    }
  }
}
```

### Training

Train models using the provided training script:

```bash
# Train models with historical data
python train.py --data historical_data.csv --config config.json --output training_results.json

# Train with verbose logging
python train.py --data data.csv --verbose

# Train for specific symbol
python train.py --data data.csv --symbol BTC/USDT
```

### Signal Filtering

Predictive models can filter signals based on predictions:

- **Price Direction Filter**: Block BUY signals if price predicts down
- **Volatility Filter**: Optionally block signals in high volatility
- **Volume Surge Filter**: Require volume surge for signal confirmation
- **Confidence Threshold**: Minimum confidence required for predictions

### Integration with Signal Pipeline

```python
# Signal processing with predictive filtering
signal_router = SignalRouter(risk_manager)

# Process signal with predictive models
approved_signal = await signal_router.process_signal(signal, market_data)

# Predictions are stored in signal metadata
if approved_signal and 'predictions' in approved_signal.metadata:
    predictions = approved_signal.metadata['predictions']
    print(f"Price direction: {predictions['price_direction']}")
    print(f"Volatility: {predictions['volatility']}")
    print(f"Volume surge: {predictions['volume_surge']}")
```

### Architecture

#### Core Components

1. **PredictiveModelManager** (`predictive_models/predictive_model_manager.py`)
   - Coordinates all predictive models
   - Manages model loading and prediction
   - Applies filtering logic

2. **Individual Predictors**
   - **PricePredictor** (`predictive_models/price_predictor.py`): Price direction classification
   - **VolatilityPredictor** (`predictive_models/volatility_predictor.py`): Volatility forecasting
   - **VolumePredictor** (`predictive_models/volume_predictor.py`): Volume surge detection

3. **PredictionContext** (`predictive_models/types.py`)
   - Container for all predictions
   - Includes confidence scores and metadata

#### Signal Processing Flow

1. **Market Data Extraction**: Extract OHLCV data from signal/market_data
2. **Model Predictions**: Generate predictions from all enabled models
3. **Signal Filtering**: Apply prediction-based filters to trading signals
4. **Metadata Storage**: Store predictions in signal metadata for later use
5. **Logging**: Log prediction results and filtering decisions

### Training and Validation

- **Cross-Validation**: Time series split for proper validation
- **Feature Engineering**: Technical indicators and statistical features
- **Model Persistence**: Save/load trained models with joblib
- **Performance Metrics**: Accuracy, R², correlation for different models

### Testing

Comprehensive test suite for predictive models:

```bash
# Run predictive model tests
python -m pytest tests/test_predictive_models.py -v

# Test specific components
python -m pytest tests/test_predictive_models.py::TestPricePredictor -v
python -m pytest tests/test_predictive_models.py::TestPredictiveModelManager -v
```

### Benefits

1. **Enhanced Signal Quality**: Filter signals based on market predictions
2. **Risk Management**: Avoid trades in adverse market conditions
3. **Adaptive Filtering**: Confidence-based signal acceptance
4. **Backtesting Integration**: Compare performance with/without filters
5. **Extensible Architecture**: Easy to add new predictive models

### Files Created/Modified

- `predictive_models/` - New module directory
  - `__init__.py` - Module initialization
  - `types.py` - PredictionContext dataclass
  - `price_predictor.py` - Price direction prediction
  - `volatility_predictor.py` - Volatility forecasting
  - `volume_predictor.py` - Volume surge detection
  - `predictive_model_manager.py` - Main coordination class
- `core/signal_router.py` - Integration with signal processing
- `utils/config_loader.py` - Configuration support
- `train.py` - Model training script
- `requirements.txt` - Added arch library for GARCH
- `config_ensemble_example.json` - Example configuration
- `tests/test_predictive_models.py` - Unit tests

## Advanced Execution Logic

The framework now includes sophisticated order execution strategies that reduce slippage, minimize market impact, and provide better average execution prices through intelligent order splitting and timing.

### Features

#### Smart Order Execution
- **Automatic Splitting**: Large orders are automatically split into smaller parts
- **Configurable Thresholds**: Define when to split orders based on size/value
- **Delay Management**: Configurable delays between order parts
- **Market Impact Reduction**: Prevents large orders from moving the market

#### TWAP (Time-Weighted Average Price)
- **Equal Distribution**: Orders split equally over specified time period
- **Predictable Execution**: Fixed intervals ensure consistent execution
- **Market Impact Control**: Reduces price impact through gradual execution
- **Schedule Management**: Pre-calculated execution schedule

#### VWAP (Volume-Weighted Average Price)
- **Liquidity Analysis**: Execution weighted by historical volume patterns
- **Optimal Timing**: Higher execution during high-volume periods
- **Volume Profile**: Analyzes historical volume distribution
- **Adaptive Weights**: Dynamic weighting based on market conditions

#### DCA (Dollar-Cost Averaging)
- **Session Management**: Multi-part orders executed over time
- **Interval Control**: Configurable time intervals between executions
- **Progress Tracking**: Real-time monitoring of DCA session progress
- **Risk Distribution**: Reduces timing risk through gradual entry/exit

### Configuration

```json
{
  "execution": {
    "enabled": true,
    "mode": "smart",
    "smart": {
      "split_threshold": 5000,
      "max_parts": 5,
      "delay_seconds": 2,
      "fallback_mode": "market"
    },
    "twap": {
      "duration_minutes": 30,
      "parts": 10,
      "fallback_mode": "market"
    },
    "vwap": {
      "lookback_minutes": 60,
      "parts": 10,
      "fallback_mode": "market"
    },
    "dca": {
      "interval_minutes": 60,
      "parts": 5,
      "fallback_mode": "market"
    }
  }
}
```

### Architecture

#### Core Components

1. **BaseExecutor** (`core/execution/base_executor.py`)
   - Abstract base class defining execution interface
   - Common functionality for order splitting and validation
   - Signal processing and metadata management

2. **SmartOrderExecutor** (`core/execution/smart_order_executor.py`)
   - Intelligent order splitting based on size thresholds
   - Delay management between order parts
   - Fallback handling for failed executions

3. **TWAPExecutor** (`core/execution/twap_executor.py`)
   - Time-based order distribution
   - Execution schedule calculation
   - Progress tracking and completion handling

4. **VWAPExecutor** (`core/execution/vwap_executor.py`)
   - Volume profile analysis
   - Liquidity-based execution weighting
   - Historical volume data processing

5. **DCAExecutor** (`core/execution/dca_executor.py`)
   - Session-based order management
   - Interval timing and scheduling
   - Multi-part execution tracking

#### Integration with OrderManager

```python
# OrderManager integration
if config.get("execution", {}).get("enabled", False):
    execution_config = config["execution"]
    mode = execution_config.get("mode", "smart")

    if mode == "smart":
        executor = SmartOrderExecutor(execution_config.get("smart", {}), exchange_api)
    elif mode == "twap":
        executor = TWAPExecutor(execution_config.get("twap", {}), exchange_api)
    elif mode == "vwap":
        executor = VWAPExecutor(execution_config.get("vwap", {}), exchange_api)
    elif mode == "dca":
        executor = DCAExecutor(execution_config.get("dca", {}), exchange_api)

    # Execute order using selected strategy
    orders = await executor.execute_order(signal)
```

### Usage Examples

#### Smart Order Execution
```python
from core.execution import SmartOrderExecutor

config = {
    "split_threshold": 10000,  # Split orders over $10k
    "max_parts": 5,
    "delay_seconds": 3
}
executor = SmartOrderExecutor(config, exchange_api)

# Large order will be automatically split
signal = TradingSignal(amount=Decimal(50), ...)  # $50k order
orders = await executor.execute_order(signal)  # Split into 5 parts
```

#### TWAP Execution
```python
from core.execution import TWAPExecutor

config = {
    "duration_minutes": 30,
    "parts": 10
}
executor = TWAPExecutor(config, exchange_api)

# Order executed over 30 minutes in 10 equal parts
signal = TradingSignal(amount=Decimal(10), ...)
orders = await executor.execute_order(signal)

# Check execution schedule
schedule = executor.get_execution_schedule()
```

#### VWAP Execution
```python
from core.execution import VWAPExecutor

config = {
    "lookback_minutes": 60,
    "parts": 10
}
executor = VWAPExecutor(config, exchange_api)

# Order weighted by historical volume profile
signal = TradingSignal(amount=Decimal(10), ...)
orders = await executor.execute_order(signal)
```

#### DCA Execution
```python
from core.execution import DCAExecutor

config = {
    "interval_minutes": 60,
    "parts": 5
}
executor = DCAExecutor(config, exchange_api)

# Start DCA session
signal = TradingSignal(amount=Decimal(5), ...)
orders = await executor.execute_order(signal)  # Executes first part

# Monitor DCA sessions
sessions = executor.get_dca_sessions()
status = executor.get_session_status(session_id)

# Cancel DCA session if needed
await executor.cancel_dca_session(session_id)
```

### Signal Processing Flow

1. **Execution Mode Selection**: Choose executor based on configuration
2. **Signal Validation**: Validate signal parameters and market conditions
3. **Order Preparation**: Split orders and prepare execution schedule
4. **Sequential Execution**: Execute order parts with proper timing
5. **Progress Tracking**: Monitor execution progress and handle failures
6. **Completion Handling**: Clean up resources and log results

### Logging and Monitoring

All execution strategies provide comprehensive logging:

```json
{
  "execution_mode": "twap",
  "symbol": "BTC/USDT",
  "total_amount": 10.0,
  "parts": 5,
  "duration_minutes": 30,
  "progress": "3/5",
  "next_execution": "2025-01-15T10:30:00Z"
}
```

### Error Handling and Fallbacks

- **Partial Failures**: Continue execution if individual parts fail
- **Fallback Modes**: Switch to market orders if advanced execution fails
- **Session Recovery**: Resume DCA sessions after restarts
- **Timeout Handling**: Cancel orders that exceed time limits

### Testing

Comprehensive test suite for execution strategies:

```bash
# Run all execution tests
python -m pytest tests/test_execution.py -v

# Test specific executors
python -m pytest tests/test_execution.py::TestSmartOrderExecutor -v
python -m pytest tests/test_execution.py::TestTWAPExecutor -v
python -m pytest tests/test_execution.py::TestDCAExecutor -v
```

### Benefits

1. **Reduced Slippage**: Intelligent order splitting minimizes price impact
2. **Better Average Prices**: TWAP and VWAP provide optimal execution prices
3. **Risk Management**: DCA distributes entry/exit risk over time
4. **Market Impact Control**: Large orders executed without moving the market
5. **Flexible Configuration**: Adaptable to different market conditions
6. **Real-time Monitoring**: Track execution progress and performance

### Files Created/Modified

- `core/execution/` - New execution module
  - `base_executor.py` - Abstract base class
  - `smart_order_executor.py` - Smart order splitting
  - `twap_executor.py` - Time-weighted execution
  - `vwap_executor.py` - Volume-weighted execution
  - `dca_executor.py` - Dollar-cost averaging
  - `__init__.py` - Module exports
- `utils/config_loader.py` - Execution configuration support
- `config_ensemble_example.json` - Example configuration
- `tests/test_execution.py` - Comprehensive unit tests

## Self-Optimization Layer

The framework now includes advanced self-optimization capabilities that automatically adapt trading strategies over time without manual retuning. The optimization layer uses multiple techniques to find optimal parameter combinations and strategy selections.

### Features

#### Walk-Forward Optimization (WFO)
- **Rolling Windows**: Splits historical data into multiple train/test windows
- **Out-of-Sample Validation**: Ensures optimization doesn't overfit to historical data
- **Automatic Updates**: Updates strategy parameters when OOS performance improves
- **Configurable Thresholds**: Minimum improvement required for parameter updates

#### Genetic Algorithm (GA) Optimization
- **Population-Based**: Maintains a population of parameter combinations
- **Evolutionary Operators**: Selection, crossover, and mutation operators
- **Fitness Evaluation**: Comprehensive fitness scoring with multiple metrics
- **Convergence Detection**: Automatic detection of optimization convergence

#### Reinforcement Learning (RL) Strategy Selector
- **Market State Representation**: Encodes market conditions (trend, volatility, volume)
- **Q-Learning**: Learns optimal strategy selection through experience
- **Action Space**: Multiple trading strategies as selectable actions
- **Reward Function**: Customizable reward based on P&L, Sharpe ratio, win rate

### Configuration

```json
{
  "optimization": {
    "enabled": true,
    "mode": "ga",
    "wfo": {
      "train_window_days": 90,
      "test_window_days": 30,
      "rolling": true,
      "min_observations": 1000,
      "improvement_threshold": 0.05
    },
    "ga": {
      "population_size": 20,
      "generations": 10,
      "mutation_rate": 0.1,
      "crossover_rate": 0.7,
      "elitism_rate": 0.1,
      "tournament_size": 3
    },
    "rl": {
      "alpha": 0.1,
      "gamma": 0.95,
      "epsilon": 0.1,
      "episodes": 100,
      "max_steps_per_episode": 50,
      "reward_function": "sharpe_ratio"
    },
    "fitness_metric": "sharpe_ratio",
    "fitness_weights": {
      "sharpe_ratio": 1.0,
      "total_return": 0.3,
      "win_rate": 0.2,
      "max_drawdown": -0.1
    }
  }
}
```

### Architecture

#### Core Components

1. **BaseOptimizer** (`optimization/base_optimizer.py`)
   - Abstract base class defining optimization interface
   - Common functionality for fitness evaluation and parameter validation
   - Results persistence and convergence detection

2. **WalkForwardOptimizer** (`optimization/walk_forward.py`)
   - Rolling window generation and management
   - Out-of-sample performance validation
   - Automatic parameter updates based on improvement thresholds

3. **GeneticOptimizer** (`optimization/genetic_optimizer.py`)
   - Population management and evolution
   - Genetic operators (selection, crossover, mutation)
   - Chromosome representation of strategy parameters

4. **RLOptimizer** (`optimization/rl_optimizer.py`)
   - Q-learning implementation for strategy selection
   - Market state representation and encoding
   - Policy learning and exploitation

5. **OptimizerFactory** (`optimization/optimizer_factory.py`)
   - Factory pattern for optimizer creation
   - Configuration validation and default settings
   - Registry of available optimization techniques

#### Parameter Bounds and Validation

```python
from optimization.base_optimizer import ParameterBounds

# Define parameter constraints
rsi_bounds = ParameterBounds(
    name='rsi_period',
    min_value=5,
    max_value=50,
    param_type='int'
)

optimizer.add_parameter_bounds(rsi_bounds)
```

### Usage Examples

#### Walk-Forward Optimization
```python
from optimization import WalkForwardOptimizer

config = {
    'train_window_days': 90,
    'test_window_days': 30,
    'rolling': True,
    'improvement_threshold': 0.05
}
optimizer = WalkForwardOptimizer(config)

# Run optimization
best_params = optimizer.optimize(strategy_class, historical_data)
print(f"Best parameters: {best_params}")
```

#### Genetic Algorithm Optimization
```python
from optimization import GeneticOptimizer

config = {
    'population_size': 20,
    'generations': 10,
    'mutation_rate': 0.1,
    'crossover_rate': 0.7
}
optimizer = GeneticOptimizer(config)

# Add parameter bounds
optimizer.add_parameter_bounds(ParameterBounds('rsi_period', 5, 50, param_type='int'))
optimizer.add_parameter_bounds(ParameterBounds('ema_fast', 5, 20, param_type='int'))

best_params = optimizer.optimize(strategy_class, historical_data)
```

#### Reinforcement Learning Strategy Selection
```python
from optimization import RLOptimizer

config = {
    'episodes': 100,
    'alpha': 0.1,
    'gamma': 0.95,
    'epsilon': 0.1
}
optimizer = RLOptimizer(config)

# Train RL agent
policy = optimizer.optimize(strategy_class, historical_data)

# Use learned policy for prediction
current_action = optimizer.predict_action(market_data)
print(f"Recommended strategy: {current_action}")
```

#### Factory Pattern Usage
```python
from optimization import OptimizerFactory

# Create optimizer from configuration
config = {
    'optimization': {
        'enabled': True,
        'mode': 'ga',
        'population_size': 20,
        'generations': 10
    }
}

optimizer = OptimizerFactory.create_from_config(config)
if optimizer:
    best_params = optimizer.optimize(strategy_class, data)
```

### Integration with Strategy Framework

The optimization layer integrates seamlessly with the existing strategy framework:

```python
# In StrategyManager or main trading loop
if config.get("optimization", {}).get("enabled", False):
    optimizer = OptimizerFactory.create_from_config(config)
    if optimizer:
        # Add parameter bounds for your strategy
        optimizer.add_parameter_bounds(ParameterBounds('rsi_period', 5, 50, 'int'))
        optimizer.add_parameter_bounds(ParameterBounds('ema_period', 5, 50, 'int'))

        # Run optimization
        best_params = optimizer.optimize(strategy_class, historical_data)

        # Update strategy with optimized parameters
        strategy.update_params(best_params)

        # Save optimization results
        optimizer.save_results("optimization_results.json")
```

### Fitness Evaluation

The optimization framework supports comprehensive fitness evaluation:

- **Primary Metrics**: Sharpe ratio, total return, win rate, max drawdown
- **Composite Scoring**: Weighted combination of multiple metrics
- **Custom Weights**: Configurable importance of different metrics
- **Backtesting Integration**: Uses existing backtesting infrastructure

### Results Persistence

All optimizers support saving and loading optimization results:

```python
# Save optimization results
optimizer.save_results("results/optimization_results.json")

# Load previous results
previous_results = optimizer.load_results("results/optimization_results.json")

# Continue optimization from previous state
if previous_results:
    optimizer.best_params = previous_results.best_params
    optimizer.best_fitness = previous_results.best_fitness
```

### Logging and Monitoring

Comprehensive logging throughout the optimization process:

```json
{
  "level": "INFO",
  "message": "Gen 5/10 | Best: 2.34 | Avg: 1.87 | Std: 0.45 | Params: {'rsi_period': 14, 'ema_period': 21}",
  "optimizer": "GeneticOptimizer",
  "generation": 5,
  "best_fitness": 2.34
}
```

### Testing

Comprehensive test suite for all optimization techniques:

```bash
# Run all optimization tests
python -m pytest tests/test_optimization.py -v

# Test specific optimizers
python -m pytest tests/test_optimization.py::TestWalkForwardOptimizer -v
python -m pytest tests/test_optimization.py::TestGeneticOptimizer -v
python -m pytest tests/test_optimization.py::TestRLOptimizer -v
```

### Performance Considerations

- **Computational Efficiency**: Configurable population sizes and generations
- **Parallel Evaluation**: Support for parallel fitness evaluation
- **Early Stopping**: Convergence detection to stop optimization early
- **Memory Management**: Efficient storage of optimization state

### Advanced Features

#### Custom Fitness Functions
```python
def custom_fitness(metrics: Dict[str, Any]) -> float:
    """Custom fitness function combining multiple metrics."""
    sharpe = metrics.get('sharpe_ratio', 0)
    return = metrics.get('total_return', 0)
    win_rate = metrics.get('win_rate', 0)

    # Custom scoring logic
    return sharpe * 0.5 + return * 0.3 + win_rate * 0.2

optimizer.fitness_function = custom_fitness
```

#### Parameter Constraints
```python
# Complex parameter relationships
def validate_params(params: Dict[str, Any]) -> bool:
    """Custom parameter validation."""
    fast_period = params.get('fast_period', 0)
    slow_period = params.get('slow_period', 0)

    # Ensure fast period is less than slow period
    return fast_period < slow_period

optimizer.parameter_validator = validate_params
```

### Benefits

1. **Automatic Adaptation**: Strategies adapt to changing market conditions
2. **Overfitting Prevention**: Out-of-sample validation prevents curve fitting
3. **Multiple Techniques**: Choice of optimization algorithms for different needs
4. **Comprehensive Evaluation**: Multi-metric fitness evaluation
5. **Production Ready**: Results persistence and policy deployment
6. **Extensible Architecture**: Easy to add new optimization techniques

### Files Created/Modified

- `optimization/` - New optimization module
  - `__init__.py` - Module initialization
  - `base_optimizer.py` - Abstract base class and utilities
  - `walk_forward.py` - Walk-forward optimization
  - `genetic_optimizer.py` - Genetic algorithm optimization
  - `rl_optimizer.py` - Reinforcement learning optimization
  - `optimizer_factory.py` - Factory for optimizer creation
- `utils/config_loader.py` - Optimization configuration support
- `config_ensemble_example.json` - Example configuration
- `tests/test_optimization.py` - Comprehensive unit tests

## Conclusion

The Spot Trading Framework now includes a comprehensive self-optimization layer that enables strategies to automatically adapt and improve over time. The modular architecture supports multiple optimization techniques and integrates seamlessly with the existing trading framework.

Combined with Ensemble Strategies, Predictive Models, and Advanced Execution, the framework provides:

- **Strategy Combination**: Ensemble methods for robust signal generation
- **Market Prediction**: ML/statistical models for market condition assessment
- **Advanced Execution**: Sophisticated order execution with minimal slippage
- **Self-Optimization**: Automatic strategy adaptation and parameter tuning
- **Risk Management**: Multi-layered filtering, validation, and position management
- **Performance Optimization**: Dynamic weights, adaptive learning, and optimal execution
- **Comprehensive Testing**: Full test coverage for reliability

## Capital & Portfolio Management Layer

The framework now includes comprehensive portfolio management capabilities for multi-asset trading, enabling intelligent capital allocation, dynamic asset rotation, and risk management across multiple cryptocurrencies.

### Features

#### Multi-Asset Position Tracking
- **Real-time Portfolio Value**: Tracks cash balance and positions across all assets
- **P&L Calculation**: Real-time unrealized and realized P&L for each position
- **Performance Metrics**: Sharpe ratio, max drawdown, win rate, and portfolio statistics
- **Allocation History**: Complete history of portfolio allocations and rebalancing

#### Dynamic Asset Rotation
- **Momentum-Based Selection**: Ranks assets by recent performance and momentum
- **Signal Strength Filtering**: Selects assets based on strategy signal strength
- **Performance-Based Rotation**: Rotates out underperforming assets automatically
- **Configurable Top-N Selection**: Choose how many assets to hold simultaneously

#### Adaptive Rebalancing
- **Threshold-Based Rebalancing**: Rebalances when allocations deviate beyond threshold
- **Periodic Rebalancing**: Scheduled rebalancing at fixed intervals
- **Multiple Allocation Schemes**: Equal weight, risk parity, momentum weighted
- **Transaction Cost Awareness**: Minimizes unnecessary trading

#### Capital Allocation Strategies
- **Equal Weight**: Equal capital allocation to all selected assets
- **Risk Parity**: Allocation based on risk contribution (lower risk = higher allocation)
- **Momentum Weighted**: Higher allocation to assets with stronger recent performance
- **Minimum Variance**: Optimization for minimum portfolio variance

#### Portfolio Hedging
- **Market Condition Monitoring**: ADX, volatility, and drawdown triggers
- **Stablecoin Rotation**: Automatic shift to stablecoins in adverse conditions
- **Partial Exit Strategy**: Reduce exposure during high volatility
- **Configurable Triggers**: Customizable hedging thresholds and conditions

### Configuration

```json
{
  "portfolio": {
    "enabled": true,
    "rotation": {
      "method": "momentum",
      "lookback_days": 30,
      "top_n": 5
    },
    "rebalancing": {
      "mode": "threshold",
      "threshold": 0.05,
      "period_days": 7,
      "scheme": "risk_parity"
    },
    "hedging": {
      "enabled": true,
      "max_stablecoin_pct": 0.3,
      "trigger": {
        "adx_below": 15,
        "volatility_above": 0.05,
        "drawdown_above": 0.1
      }
    },
    "allocation": {
      "min_position_size": 100.0,
      "max_position_size": 10000.0,
      "max_assets": 10,
      "risk_per_asset": 0.02
    }
  }
}
```

### Architecture

#### Core Components

1. **PortfolioManager** (`portfolio/portfolio_manager.py`)
   - Main portfolio management orchestrator
   - Position tracking and P&L calculation
   - Asset rotation and rebalancing coordination
   - Portfolio metrics and reporting

2. **CapitalAllocator** (`portfolio/allocator.py`)
   - Abstract base class for allocation strategies
   - Equal weight, risk parity, momentum weighted implementations
   - Allocation validation and normalization

3. **PortfolioHedger** (`portfolio/hedging.py`)
   - Hedging strategy implementation
   - Market condition analysis and trigger detection
   - Hedging action execution and monitoring

#### Position Management

```python
from portfolio.portfolio_manager import Position

# Create and manage positions
position = Position(
    symbol='BTC/USDT',
    quantity=Decimal('1.5'),
    entry_price=Decimal('45000'),
    current_price=Decimal('46000')
)

# Automatic P&L calculation
position.update_pnl(Decimal('46500'))
print(f"Unrealized P&L: {position.unrealized_pnl}")
print(f"Total P&L: {position.total_pnl}")
```

### Usage Examples

#### Basic Portfolio Management
```python
from portfolio import PortfolioManager

config = {
    'rotation': {'method': 'momentum', 'top_n': 5},
    'rebalancing': {'mode': 'threshold', 'scheme': 'equal_weight'},
    'hedging': {'enabled': True}
}

portfolio_manager = PortfolioManager(config, initial_balance=Decimal('50000'))

# Update prices
price_data = {
    'BTC/USDT': Decimal('45000'),
    'ETH/USDT': Decimal('3000'),
    'ADA/USDT': Decimal('1.2')
}
portfolio_manager.update_prices(price_data)

# Get portfolio metrics
metrics = portfolio_manager.get_portfolio_metrics()
print(f"Portfolio Value: ${metrics.total_value}")
print(f"Total P&L: ${metrics.total_pnl}")
```

#### Asset Rotation
```python
# Market data for momentum calculation
market_data = pd.DataFrame({
    'BTC/USDT': historical_prices_btc,
    'ETH/USDT': historical_prices_eth,
    'ADA/USDT': historical_prices_ada
})

# Strategy signals per asset
strategy_signals = {
    'BTC/USDT': [{'signal_strength': 0.8}],
    'ETH/USDT': [{'signal_strength': 0.6}],
    'ADA/USDT': [{'signal_strength': 0.4}]
}

# Rotate to top performing assets
selected_assets = portfolio_manager.rotate_assets(strategy_signals, market_data)
print(f"Selected assets: {selected_assets}")
```

#### Rebalancing
```python
from portfolio.allocator import EqualWeightAllocator

# Create allocator
allocator = EqualWeightAllocator()

# Get target allocations
target_allocations = allocator.allocate(selected_assets)

# Rebalance portfolio
result = portfolio_manager.rebalance(target_allocations, current_prices)

if result['rebalanced']:
    print(f"Executed {len(result['trades'])} rebalancing trades")
    for trade in result['trades']:
        print(f"{trade['side']} {trade['quantity']} {trade['symbol']}")
```

#### Hedging
```python
# Market conditions for hedging
market_conditions = {
    'adx': 12,  # Below threshold
    'volatility': 0.08,  # High volatility
    'drawdown': 0.05  # Portfolio drawdown
}

# Evaluate hedging
hedge_actions = portfolio_manager.hedge_positions(market_conditions)

if hedge_actions:
    print(f"Hedging triggered: {hedge_actions['trigger_reason']}")
    print(f"Hedging trades: {len(hedge_actions['executed_trades'])}")
```

### Allocation Strategies

#### Equal Weight Allocation
```python
from portfolio.allocator import EqualWeightAllocator

allocator = EqualWeightAllocator()
allocations = allocator.allocate(['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
# Result: {'BTC/USDT': 0.333, 'ETH/USDT': 0.333, 'ADA/USDT': 0.333}
```

#### Risk Parity Allocation
```python
from portfolio.allocator import RiskParityAllocator

allocator = RiskParityAllocator({'lookback_period': 30})
allocations = allocator.allocate(assets, market_data)
# Higher volatility assets get lower allocation
```

#### Momentum Weighted Allocation
```python
from portfolio.allocator import MomentumWeightAllocator

allocator = MomentumWeightAllocator({'lookback_period': 30})
allocations = allocator.allocate(assets, market_data)
# Stronger momentum assets get higher allocation
```

### Integration with Strategy Framework

The portfolio layer integrates seamlessly with the existing strategy framework:

```python
# In main trading loop or strategy manager
if config.get("portfolio", {}).get("enabled", False):
    # Get strategy signals for all assets
    strategy_signals = strategy_manager.get_all_signals()

    # Rotate to best performing assets
    selected_assets = portfolio_manager.rotate_assets(strategy_signals, market_data)

    # Allocate capital using selected strategy
    if config["portfolio"]["rebalancing"]["scheme"] == "equal_weight":
        allocator = EqualWeightAllocator()
    elif config["portfolio"]["rebalancing"]["scheme"] == "risk_parity":
        allocator = RiskParityAllocator()
    # ... other allocators

    target_allocations = allocator.allocate(selected_assets, market_data)

    # Rebalance portfolio
    rebalance_result = portfolio_manager.rebalance(target_allocations, current_prices)

    # Apply hedging if needed
    hedge_result = portfolio_manager.hedge_positions(market_conditions)

    # Log portfolio status
    metrics = portfolio_manager.get_portfolio_metrics()
    logger.info(f"Portfolio Value: ${metrics.total_value}, P&L: ${metrics.total_pnl}")
```

### Performance Metrics

The portfolio manager provides comprehensive performance tracking:

```python
metrics = portfolio_manager.get_portfolio_metrics()

print(f"Total Value: ${metrics.total_value}")
print(f"Total P&L: ${metrics.total_pnl}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio}")
print(f"Max Drawdown: {metrics.max_drawdown * 100}%")
print(f"Win Rate: {metrics.win_rate * 100}%")
print(f"Number of Positions: {metrics.num_positions}")
print(f"Number of Assets: {metrics.num_assets}")
```

### Export and Reporting

```python
# Export allocation history
portfolio_manager.export_allocation_history("portfolio_allocations.csv")

# Get detailed portfolio summary
summary = portfolio_manager.get_portfolio_summary()
print(json.dumps(summary, indent=2, default=str))
```

### Testing

Comprehensive test suite for portfolio management:

```bash
# Run all portfolio tests
python -m pytest tests/test_portfolio.py -v

# Test specific components
python -m pytest tests/test_portfolio.py::TestPortfolioManager -v
python -m pytest tests/test_portfolio.py::TestCapitalAllocators -v
python -m pytest tests/test_portfolio.py::TestPortfolioHedger -v
```

### Risk Management Features

- **Position Size Limits**: Minimum and maximum position sizes
- **Asset Count Limits**: Maximum number of assets to hold
- **Risk per Asset**: Maximum risk allocation per individual asset
- **Hedging Triggers**: Automatic hedging based on market conditions
- **Drawdown Protection**: Reduce exposure during portfolio drawdowns

### Advanced Features

#### Custom Allocation Strategies
```python
class CustomAllocator(CapitalAllocator):
    def allocate(self, assets, market_data=None, current_prices=None):
        # Implement custom allocation logic
        allocations = {}
        for asset in assets:
            # Custom logic here
            allocations[asset] = custom_weight
        return allocations
```

#### Dynamic Hedging Rules
```python
# Custom hedging conditions
custom_triggers = {
    'custom_indicator': lambda x: x < threshold,
    'portfolio_heat': lambda x: x > max_heat
}

hedger = PortfolioHedger({
    'enabled': True,
    'custom_triggers': custom_triggers
})
```

### Benefits

1. **Multi-Asset Diversification**: Intelligent allocation across multiple cryptocurrencies
2. **Dynamic Adaptation**: Automatic rotation to best performing assets
3. **Risk Management**: Hedging and position size controls
4. **Performance Optimization**: Multiple allocation strategies for different market conditions
5. **Real-time Monitoring**: Comprehensive portfolio metrics and reporting
6. **Flexible Configuration**: Customizable parameters for all features
7. **Backtesting Integration**: Historical performance analysis and optimization

### Files Created/Modified

- `portfolio/` - New portfolio management module
  - `__init__.py` - Module initialization
  - `portfolio_manager.py` - Main portfolio management class
  - `allocator.py` - Capital allocation strategies
  - `hedging.py` - Portfolio hedging functionality
- `utils/config_loader.py` - Portfolio configuration support
- `config_ensemble_example.json` - Example configuration
- `tests/test_portfolio.py` - Comprehensive unit tests

## Conclusion

The Spot Trading Framework now includes a comprehensive Capital & Portfolio Management Layer that enables sophisticated multi-asset trading strategies. The modular architecture supports various allocation schemes, dynamic asset rotation, and risk management through hedging.

Combined with Ensemble Strategies, Predictive Models, Advanced Execution, and Self-Optimization, the framework provides:

- **Strategy Combination**: Ensemble methods for robust signal generation
- **Market Prediction**: ML/statistical models for market condition assessment
- **Advanced Execution**: Sophisticated order execution with minimal slippage
- **Self-Optimization**: Automatic strategy adaptation and parameter tuning
- **Portfolio Management**: Multi-asset capital allocation and risk management
- **Risk Management**: Multi-layered filtering, validation, and position management
- **Performance Optimization**: Dynamic weights, adaptive learning, and optimal execution
- **Comprehensive Testing**: Full test coverage for reliability

The implementation maintains backward compatibility while providing enterprise-grade portfolio management capabilities for professional multi-asset trading operations. The portfolio layer can be enabled or disabled based on user preferences and seamlessly integrates with existing single-asset strategies.
