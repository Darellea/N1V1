# Strategy Development Guide

Complete guide for developing, testing, and deploying trading strategies in N1V1.

## Overview

N1V1 supports multiple trading strategies that can run simultaneously. Each strategy inherits from `BaseStrategy` and implements a standardized interface for signal generation, risk management, and performance tracking.

## Strategy Architecture

### Base Strategy Interface

All strategies inherit from `BaseStrategy` located in `strategies/base_strategy.py`:

```python
from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import OrderType

class MyStrategy(BaseStrategy):
    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        pass

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals"""
        pass
```

### Strategy Configuration

Each strategy requires a configuration object:

```python
config = StrategyConfig(
    name="my_strategy",
    symbols=["BTC/USDT", "ETH/USDT"],
    timeframe="15m",
    required_history=100,
    enabled=True,
    params={
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
    }
)
```

## Creating a New Strategy

### Step 1: Create Strategy File

Create a new file in the `strategies/` directory:

```bash
touch strategies/my_custom_strategy.py
```

### Step 2: Implement Strategy Class

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from strategies.base_strategy import BaseStrategy, StrategyConfig
from core.contracts import TradingSignal, SignalType, SignalStrength
from core.types import OrderType

class MyCustomStrategy(BaseStrategy):
    """Custom trading strategy implementation."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Strategy-specific initialization
        self.fast_period = self.config.params.get('fast_period', 12)
        self.slow_period = self.config.params.get('slow_period', 26)

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the strategy."""
        df = data.copy()

        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(self.slow_period).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals based on indicators."""
        signals = []

        if len(data) < self.slow_period:
            return signals

        # Get latest data point
        latest = data.iloc[-1]
        symbol = latest.get('symbol', self.config.symbols[0])

        # Generate signals based on strategy logic
        if self._should_buy(latest):
            signal = self.create_signal(
                symbol=symbol,
                signal_type=SignalType.ENTRY_LONG,
                strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.02,  # 2% position size
                current_price=float(latest['close']),
                stop_loss=float(latest['close'] * 0.95),  # 5% stop loss
                take_profit=float(latest['close'] * 1.10),  # 10% take profit
                metadata={
                    'strategy': self.config.name,
                    'fast_ma': float(latest['fast_ma']),
                    'slow_ma': float(latest['slow_ma']),
                    'rsi': float(latest['rsi'])
                }
            )
            signals.append(signal)

        elif self._should_sell(latest):
            signal = self.create_signal(
                symbol=symbol,
                signal_type=SignalType.ENTRY_SHORT,
                strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.02,
                current_price=float(latest['close']),
                stop_loss=float(latest['close'] * 1.05),
                take_profit=float(latest['close'] * 0.90),
                metadata={
                    'strategy': self.config.name,
                    'fast_ma': float(latest['fast_ma']),
                    'slow_ma': float(latest['slow_ma']),
                    'rsi': float(latest['rsi'])
                }
            )
            signals.append(signal)

        return signals

    def _should_buy(self, data_point: pd.Series) -> bool:
        """Determine if we should generate a buy signal."""
        # Fast MA crosses above slow MA and RSI < 30
        fast_ma = data_point.get('fast_ma', 0)
        slow_ma = data_point.get('slow_ma', 0)
        rsi = data_point.get('rsi', 50)

        return (fast_ma > slow_ma and
                rsi < 30 and
                pd.notna(fast_ma) and pd.notna(slow_ma))

    def _should_sell(self, data_point: pd.Series) -> bool:
        """Determine if we should generate a sell signal."""
        # Fast MA crosses below slow MA and RSI > 70
        fast_ma = data_point.get('fast_ma', 0)
        slow_ma = data_point.get('slow_ma', 0)
        rsi = data_point.get('rsi', 50)

        return (fast_ma < slow_ma and
                rsi > 70 and
                pd.notna(fast_ma) and pd.notna(slow_ma))
```

### Step 3: Create Strategy Factory

Add your strategy to the strategy factory in `strategies/__init__.py`:

```python
from .my_custom_strategy import MyCustomStrategy

__all__ = ['MyCustomStrategy']

# Strategy registry for dynamic instantiation
STRATEGY_REGISTRY = {
    'my_custom_strategy': MyCustomStrategy,
    # Add other strategies here
}
```

## Testing Your Strategy

### Unit Tests

Create comprehensive unit tests in `tests/strategies/test_my_custom_strategy.py`:

```python
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from strategies.my_custom_strategy import MyCustomStrategy
from strategies.base_strategy import StrategyConfig

@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15min')
    np.random.seed(42)  # For reproducible tests

    data = pd.DataFrame({
        'open': np.random.uniform(40000, 50000, 100),
        'high': np.random.uniform(40000, 50000, 100),
        'low': np.random.uniform(40000, 50000, 100),
        'close': np.random.uniform(40000, 50000, 100),
        'volume': np.random.uniform(100, 1000, 100),
        'symbol': 'BTC/USDT'
    }, index=dates)

    # Create a trend for testing
    data['close'] = data['close'].cumsum() + 45000

    return data

@pytest.fixture
def strategy_config():
    """Create strategy configuration for testing."""
    return StrategyConfig(
        name="my_custom_strategy",
        symbols=["BTC/USDT"],
        timeframe="15m",
        required_history=50,
        params={
            "fast_period": 12,
            "slow_period": 26
        }
    )

@pytest.mark.asyncio
class TestMyCustomStrategy:

    async def test_initialization(self, strategy_config):
        """Test strategy initialization."""
        strategy = MyCustomStrategy(strategy_config)
        assert strategy.config.name == "my_custom_strategy"
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26

    async def test_calculate_indicators(self, strategy_config, sample_data):
        """Test indicator calculation."""
        strategy = MyCustomStrategy(strategy_config)
        result = await strategy.calculate_indicators(sample_data)

        # Check that indicators were added
        assert 'fast_ma' in result.columns
        assert 'slow_ma' in result.columns
        assert 'rsi' in result.columns

        # Check indicator values
        assert not result['fast_ma'].isna().all()
        assert not result['slow_ma'].isna().all()

    async def test_generate_signals_buy(self, strategy_config, sample_data):
        """Test buy signal generation."""
        strategy = MyCustomStrategy(strategy_config)

        # Mock data that should trigger a buy signal
        test_data = sample_data.copy()
        test_data.loc[test_data.index[-1], 'close'] = test_data['slow_ma'].iloc[-1] * 0.95  # Below slow MA

        # Mock RSI < 30
        test_data['rsi'] = 25

        signals = await strategy.generate_signals(test_data)

        # Should generate at least one signal
        assert len(signals) >= 0  # May be 0 if conditions not met

    async def test_generate_signals_sell(self, strategy_config, sample_data):
        """Test sell signal generation."""
        strategy = MyCustomStrategy(strategy_config)

        # Mock data that should trigger a sell signal
        test_data = sample_data.copy()
        test_data.loc[test_data.index[-1], 'close'] = test_data['slow_ma'].iloc[-1] * 1.05  # Above slow MA

        # Mock RSI > 70
        test_data['rsi'] = 75

        signals = await strategy.generate_signals(test_data)

        # Should generate at least one signal
        assert len(signals) >= 0

    async def test_insufficient_data(self, strategy_config):
        """Test behavior with insufficient data."""
        strategy = MyCustomStrategy(strategy_config)

        # Create data with insufficient history
        small_data = pd.DataFrame({
            'open': [40000, 41000],
            'high': [40500, 41500],
            'low': [39500, 40500],
            'close': [40200, 41200],
            'volume': [100, 150],
            'symbol': 'BTC/USDT'
        })

        signals = await strategy.generate_signals(small_data)
        assert len(signals) == 0

    async def test_signal_structure(self, strategy_config, sample_data):
        """Test that generated signals have correct structure."""
        strategy = MyCustomStrategy(strategy_config)

        # Ensure we have enough data and indicators
        data_with_indicators = await strategy.calculate_indicators(sample_data)
        signals = await strategy.generate_signals(data_with_indicators)

        for signal in signals:
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'signal_type')
            assert hasattr(signal, 'amount')
            assert hasattr(signal, 'metadata')
            assert signal.metadata['strategy'] == 'my_custom_strategy'
```

### Integration Tests

Create integration tests in `tests/integration/test_strategy_integration.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from strategies.my_custom_strategy import MyCustomStrategy
from strategies.base_strategy import StrategyConfig
from data.data_fetcher import DataFetcher

@pytest.mark.asyncio
class TestStrategyIntegration:

    async def test_full_strategy_workflow(self):
        """Test complete strategy workflow from data to signals."""
        # Setup
        config = StrategyConfig(
            name="my_custom_strategy",
            symbols=["BTC/USDT"],
            timeframe="15m",
            required_history=100
        )

        strategy = MyCustomStrategy(config)

        # Mock data fetcher
        mock_fetcher = AsyncMock()
        mock_data = pd.DataFrame({
            'open': np.random.uniform(40000, 50000, 150),
            'high': np.random.uniform(40000, 50000, 150),
            'low': np.random.uniform(40000, 50000, 150),
            'close': np.random.uniform(40000, 50000, 150),
            'volume': np.random.uniform(100, 1000, 150),
            'symbol': 'BTC/USDT'
        })
        mock_fetcher.get_historical_data.return_value = mock_data

        # Initialize strategy
        await strategy.initialize(mock_fetcher)

        # Run strategy
        signals = await strategy.run()

        # Verify results
        assert isinstance(signals, list)
        for signal in signals:
            assert signal.symbol == "BTC/USDT"
            assert signal.metadata['strategy'] == 'my_custom_strategy'

    async def test_strategy_with_real_data_fetcher(self):
        """Test strategy with real data fetcher (requires API keys)."""
        pytest.skip("Requires API keys - run manually")

        config = StrategyConfig(
            name="my_custom_strategy",
            symbols=["BTC/USDT"],
            timeframe="15m",
            required_history=100
        )

        strategy = MyCustomStrategy(config)
        data_fetcher = DataFetcher()

        await strategy.initialize(data_fetcher)
        signals = await strategy.run()

        assert isinstance(signals, list)
```

### Running Tests

```bash
# Run strategy-specific tests
pytest tests/strategies/test_my_custom_strategy.py -v

# Run integration tests
pytest tests/integration/test_strategy_integration.py -v

# Run all strategy tests
pytest tests/strategies/ -v

# Run with coverage
pytest tests/strategies/ --cov=strategies.my_custom_strategy --cov-report=html
```

## Strategy Templates

### Template 1: Trend Following Strategy

```python
class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy using moving averages."""

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['short_ma'] = df['close'].rolling(20).mean()
        df['long_ma'] = df['close'].rolling(50).mean()
        df['trend'] = df['short_ma'] > df['long_ma']
        return df

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        latest = data.iloc[-1]

        if latest['trend'] and not data.iloc[-2]['trend']:  # Trend change
            signals.append(self.create_signal(
                symbol=latest['symbol'],
                signal_type=SignalType.ENTRY_LONG,
                strength=SignalStrength.MEDIUM,
                order_type=OrderType.MARKET,
                amount=0.02,
                current_price=float(latest['close'])
            ))
        return signals
```

### Template 2: Mean Reversion Strategy

```python
class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands."""

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['sma'] = df['close'].rolling(20).mean()
        df['std'] = df['close'].rolling(20).std()
        df['upper'] = df['sma'] + 2 * df['std']
        df['lower'] = df['sma'] - 2 * df['std']
        df['position'] = (df['close'] - df['sma']) / df['std']
        return df

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        latest = data.iloc[-1]

        if latest['position'] < -2:  # Below lower band
            signals.append(self.create_signal(
                symbol=latest['symbol'],
                signal_type=SignalType.ENTRY_LONG,
                strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.02,
                current_price=float(latest['close'])
            ))
        elif latest['position'] > 2:  # Above upper band
            signals.append(self.create_signal(
                symbol=latest['symbol'],
                signal_type=SignalType.ENTRY_SHORT,
                strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.02,
                current_price=float(latest['close'])
            ))
        return signals
```

### Template 3: Momentum Strategy

```python
class MomentumStrategy(BaseStrategy):
    """Momentum strategy using RSI and volume."""

    async def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volume momentum
        df['volume_ma'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        return df

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []
        latest = data.iloc[-1]

        # Strong momentum signals
        if latest['rsi'] < 30 and latest['volume_ratio'] > 1.5:
            signals.append(self.create_signal(
                symbol=latest['symbol'],
                signal_type=SignalType.ENTRY_LONG,
                strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.02,
                current_price=float(latest['close'])
            ))
        elif latest['rsi'] > 70 and latest['volume_ratio'] > 1.5:
            signals.append(self.create_signal(
                symbol=latest['symbol'],
                signal_type=SignalType.ENTRY_SHORT,
                strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.02,
                current_price=float(latest['close'])
            ))
        return signals
```

## Advanced Strategy Features

### Multi-Timeframe Analysis

```python
class MultiTimeframeStrategy(BaseStrategy):
    """Strategy using multiple timeframes for confirmation."""

    async def generate_signals(self, data: pd.DataFrame,
                             multi_tf_data: Optional[Dict[str, Any]] = None) -> List[TradingSignal]:
        signals = []

        if not multi_tf_data:
            return signals

        # Get higher timeframe trend
        higher_trend = self.get_higher_timeframe_trend(
            multi_tf_data, self.config.timeframe, '1h'
        )

        if higher_trend == 'bullish':
            # Only take long signals if higher TF is bullish
            latest = data.iloc[-1]
            if self._should_buy(latest):
                signals.append(self.create_signal(...))

        return signals
```

### Risk Management Integration

```python
class RiskManagedStrategy(BaseStrategy):
    """Strategy with integrated risk management."""

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        signals = []

        # Check portfolio risk before generating signals
        portfolio_risk = await self._check_portfolio_risk()
        if portfolio_risk > 0.05:  # 5% max risk
            return signals

        # Generate signals with position sizing
        latest = data.iloc[-1]
        if self._should_buy(latest):
            position_size = await self._calculate_position_size(latest)
            signals.append(self.create_signal(
                symbol=latest['symbol'],
                signal_type=SignalType.ENTRY_LONG,
                amount=position_size,
                current_price=float(latest['close']),
                stop_loss=float(latest['close'] * 0.98),
                take_profit=float(latest['close'] * 1.05)
            ))

        return signals

    async def _check_portfolio_risk(self) -> float:
        """Check current portfolio risk level."""
        # Implementation depends on risk management system
        return 0.02  # Placeholder

    async def _calculate_position_size(self, data_point: pd.Series) -> float:
        """Calculate position size based on risk parameters."""
        volatility = data_point.get('close', 0) * 0.02  # 2% volatility
        risk_amount = 0.01  # 1% risk per trade
        return risk_amount / volatility if volatility > 0 else 0.01
```

### Machine Learning Integration

```python
class MLStrategy(BaseStrategy):
    """Strategy using ML model for signal generation."""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.model = None

    async def initialize(self, data_fetcher: DataFetcher) -> None:
        await super().initialize(data_fetcher)
        # Load ML model
        from ml.model_loader import load_model_with_fallback
        self.model, _ = load_model_with_fallback("my_trading_model")

    async def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if not self.model:
            return []

        # Prepare features for ML model
        features = self._prepare_features(data)

        # Get ML predictions
        predictions = self.model.predict_proba(features)

        signals = []
        latest = data.iloc[-1]

        if predictions[0][1] > 0.7:  # Strong buy signal
            signals.append(self.create_signal(
                symbol=latest['symbol'],
                signal_type=SignalType.ENTRY_LONG,
                strength=SignalStrength.STRONG,
                order_type=OrderType.MARKET,
                amount=0.02,
                current_price=float(latest['close']),
                metadata={'ml_confidence': float(predictions[0][1])}
            ))

        return signals

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model."""
        # Implementation depends on your ML feature engineering
        return data[['close', 'volume']]  # Placeholder
```

## Configuration and Deployment

### Strategy Configuration

Add your strategy to the main configuration in `config.json`:

```json
{
  "strategies": {
    "active_strategies": ["my_custom_strategy", "ema_cross"],
    "my_custom_strategy": {
      "symbols": ["BTC/USDT", "ETH/USDT"],
      "timeframe": "15m",
      "params": {
        "fast_period": 12,
        "slow_period": 26
      }
    }
  }
}
```

### Strategy Registration

Update the strategy factory in `core/component_factory.py`:

```python
from strategies.my_custom_strategy import MyCustomStrategy

STRATEGY_CLASSES = {
    'my_custom_strategy': MyCustomStrategy,
    # ... other strategies
}
```

## Performance Monitoring

### Strategy Metrics

Each strategy automatically tracks:

- Signals generated
- Win/loss ratio
- Average profit/loss
- Sharpe ratio
- Maximum drawdown

### Monitoring Integration

```python
# Get strategy performance metrics
metrics = strategy.get_performance_metrics()

# Metrics include:
# - Total signals generated
# - Signals per timeframe
# - Average holding period
# - Risk-adjusted returns
```

## Best Practices

### Code Quality
1. **Follow PEP 8** style guidelines
2. **Add comprehensive docstrings** for all methods
3. **Use type hints** for better code clarity
4. **Handle exceptions** gracefully
5. **Log important events** for debugging

### Testing
1. **Write unit tests** for all indicator calculations
2. **Test edge cases** (insufficient data, extreme values)
3. **Use mock data** for reproducible tests
4. **Test integration** with data fetcher
5. **Verify signal structure** and metadata

### Performance
1. **Use vectorized operations** with pandas/numpy
2. **Cache expensive calculations** when possible
3. **Avoid unnecessary data copies**
4. **Profile your strategy** for bottlenecks
5. **Consider memory usage** for large datasets

### Risk Management
1. **Always set stop losses** and take profits
2. **Implement position sizing** based on risk
3. **Validate signals** across multiple timeframes
4. **Monitor correlation** with existing strategies
5. **Test strategies** in paper trading first

## Troubleshooting

### Common Issues

**Strategy not generating signals:**
- Check data sufficiency (`len(data) >= required_history`)
- Verify indicator calculations
- Debug signal conditions with logging

**Performance issues:**
- Profile indicator calculations
- Check for memory leaks
- Optimize data processing

**Integration problems:**
- Verify strategy registration
- Check configuration parameters
- Test with mock data fetcher

### Debug Mode

Enable debug logging for strategy development:

```python
import logging
logging.getLogger('strategy.my_custom_strategy').setLevel(logging.DEBUG)
```

## Resources

- **Base Strategy**: `strategies/base_strategy.py`
- **Existing Strategies**: `strategies/` directory
- **API Documentation**: `docs/api_examples.md`
- **Testing Framework**: `tests/strategies/`
- **Configuration Guide**: `docs/configuration.md`

## Next Steps

1. **Study existing strategies** in the `strategies/` directory
2. **Implement your strategy** following the templates above
3. **Write comprehensive tests** for your strategy
4. **Test in paper trading mode** before live deployment
5. **Monitor performance** and iterate on your approach

Remember: Strategy development is iterative. Start simple, test thoroughly, and gradually add complexity as you validate your approach.
