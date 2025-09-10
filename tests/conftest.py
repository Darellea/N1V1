"""
Comprehensive test configuration and fixtures for N1V1 trading framework.

This module provides shared test fixtures, utilities, and configuration
for testing all four major features: Multi-Timeframe Analysis,
Predictive Regime Forecasting, Hybrid AI Strategy Generator, and
Self-Healing Engine.

Test Categories:
- Unit Tests: Individual component testing
- Integration Tests: Feature interaction testing
- Performance Tests: Latency and throughput validation
- Reliability Tests: Long-running stability testing
- End-to-End Tests: Complete workflow validation
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
from unittest.mock import Mock, AsyncMock
import tempfile
import os
from pathlib import Path

from utils.config_loader import ConfigLoader
from data.data_fetcher import DataFetcher
from strategies.base_strategy import BaseStrategy
from core.bot_engine import BotEngine
from core.order_manager import OrderManager
from core.signal_router import SignalRouter
from core.timeframe_manager import TimeframeManager
from strategies.regime.strategy_selector import StrategySelector
from strategies.regime.market_regime import MarketRegimeDetector
from optimization.strategy_generator import StrategyGenerator
from core.self_healing_engine import SelfHealingEngine


# Test Configuration
@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Load test configuration with isolated settings."""
    config = {
        "environment": {
            "mode": "backtest",
            "test_mode": True
        },
        "exchange": {
            "name": "test_exchange",
            "base_currency": "USDT",
            "markets": ["BTC/USDT", "ETH/USDT"],
            "testnet": True
        },
        "trading": {
            "portfolio_mode": True,
            "initial_balance": 10000.0,
            "max_position_size": 0.1,
            "risk_per_trade": 0.02
        },
        "strategies": {
            "active_strategies": ["rsi_strategy", "macd_strategy"],
            "strategy_config": {
                "rsi_strategy": {"period": 14, "overbought": 70, "oversold": 30},
                "macd_strategy": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
            }
        },
        "risk_management": {
            "max_drawdown": 0.1,
            "max_daily_loss": 0.05,
            "position_sizing": "fixed_percentage",
            "stop_loss_percentage": 0.02
        },
        "multi_timeframe": {
            "enabled": True,
            "timeframes": ["5m", "15m", "1h", "4h"],
            "max_history_days": 30
        },
        "regime_forecasting": {
            "enabled": True,
            "model_path": "models/test_regime_forecaster",
            "forecast_horizon": 24,
            "confidence_threshold": 0.7
        },
        "strategy_generator": {
            "enabled": True,
            "population_size": 10,  # Smaller for testing
            "generations": 5,
            "mutation_rate": 0.1
        },
        "self_healing": {
            "enabled": True,
            "heartbeat_interval": 5,  # Faster for testing
            "failure_detection_threshold": 2.0
        },
        "notifications": {
            "discord": {
                "enabled": False,  # Disabled for testing
                "webhook_url": "https://discord.com/api/webhooks/test"
            }
        },
        "monitoring": {
            "terminal_display": False,
            "update_interval": 1.0,
            "performance_tracking": True
        },
        "backtesting": {
            "timeframe": "1h",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }
    }
    return config


@pytest.fixture
def temp_dir() -> str:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_exchange():
    """Mock exchange for testing."""
    exchange = Mock()
    exchange.name = "test_exchange"
    exchange.base_currency = "USDT"
    exchange.markets = ["BTC/USDT", "ETH/USDT"]

    # Mock market data methods
    exchange.get_ticker = AsyncMock(return_value={
        'symbol': 'BTC/USDT',
        'last': 50000.0,
        'bid': 49950.0,
        'ask': 50050.0,
        'volume': 1000.0
    })

    exchange.get_balance = AsyncMock(return_value={
        'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
        'BTC': {'free': 0.0, 'used': 0.0, 'total': 0.0}
    })

    exchange.create_order = AsyncMock(return_value={
        'id': 'test_order_123',
        'symbol': 'BTC/USDT',
        'type': 'limit',
        'side': 'buy',
        'amount': 0.001,
        'price': 50000.0,
        'status': 'filled'
    })

    exchange.cancel_order = AsyncMock(return_value=True)
    exchange.get_open_orders = AsyncMock(return_value=[])
    exchange.get_closed_orders = AsyncMock(return_value=[])

    return exchange


@pytest.fixture
def mock_data_fetcher(mock_exchange):
    """Mock data fetcher for testing."""
    fetcher = Mock(spec=DataFetcher)
    fetcher.exchange = mock_exchange

    # Mock data fetching methods
    fetcher.get_historical_data = AsyncMock()
    fetcher.get_realtime_data = AsyncMock()
    fetcher.get_multiple_historical_data = AsyncMock()
    fetcher.initialize = AsyncMock()
    fetcher.close = AsyncMock()

    return fetcher


@pytest.fixture
def synthetic_market_data():
    """Generate synthetic market data for testing."""
    np.random.seed(42)  # For reproducible tests

    # Generate 1000 data points (approximately 40 days of 1h data)
    n_points = 1000
    timestamps = pd.date_range(
        start=datetime(2024, 1, 1),
        periods=n_points,
        freq='1H'
    )

    # Generate realistic price data with trends and volatility
    base_price = 50000.0
    drift = 0.0001  # Small upward drift
    volatility = 0.02  # 2% daily volatility

    # Create price series
    price_changes = np.random.normal(drift, volatility/np.sqrt(24), n_points)
    prices = base_price * np.exp(np.cumsum(price_changes))

    # Add some regime-like behavior
    for i in range(0, n_points, 200):  # Every ~8 days
        if np.random.random() > 0.4:  # 60% chance of trend
            trend_length = np.random.randint(48, 120)  # 2-5 days
            trend_strength = np.random.normal(0.001, 0.0005)
            end_idx = min(i + trend_length, n_points)
            trend_changes = np.linspace(0, trend_strength * trend_length, end_idx - i)
            prices[i:end_idx] *= np.exp(trend_changes)

    # Create OHLCV data
    high_mult = 1 + np.random.uniform(0, 0.008, n_points)
    low_mult = 1 - np.random.uniform(0, 0.008, n_points)
    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    volumes = np.random.uniform(100, 1000, n_points)

    data = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_prices,
        'high': close_prices * high_mult,
        'low': close_prices * low_mult,
        'close': close_prices,
        'volume': volumes
    })

    data.set_index('timestamp', inplace=True)

    # Set regime type attribute for test compatibility
    data.attrs['regime_type'] = 'bull_market'

    return data


@pytest.fixture
def multi_timeframe_data(synthetic_market_data):
    """Generate multi-timeframe test data."""
    data = synthetic_market_data

    # Resample to different timeframes
    timeframes = {
        '5m': data.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),

        '15m': data.resample('15min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(),

        '1h': data,  # Original 1h data

        '4h': data.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
    }

    return timeframes


@pytest.fixture
def mock_strategy():
    """Mock trading strategy for testing."""
    strategy = Mock(spec=BaseStrategy)
    strategy.name = "test_strategy"
    strategy.generate_signals = AsyncMock(return_value=[])
    strategy.initialize = AsyncMock()
    strategy.shutdown = AsyncMock()
    strategy.validate_config = Mock(return_value=True)

    return strategy


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager for testing."""
    risk_manager = Mock()
    risk_manager.evaluate_signal = AsyncMock(return_value=True)
    risk_manager.check_position_size = Mock(return_value=0.001)
    risk_manager.calculate_stop_loss = Mock(return_value=49000.0)
    risk_manager.get_max_drawdown = Mock(return_value=0.05)

    return risk_manager


@pytest.fixture
def mock_order_manager(mock_exchange):
    """Mock order manager for testing."""
    order_manager = Mock(spec=OrderManager)
    order_manager.exchange = mock_exchange
    order_manager.execute_order = AsyncMock(return_value={
        'id': 'test_order_123',
        'status': 'filled',
        'amount': 0.001,
        'price': 50000.0,
        'pnl': 0.0
    })
    order_manager.get_balance = AsyncMock(return_value=10000.0)
    order_manager.get_equity = AsyncMock(return_value=10000.0)
    order_manager.get_active_order_count = AsyncMock(return_value=0)
    order_manager.get_open_position_count = AsyncMock(return_value=0)
    order_manager.cancel_all_orders = AsyncMock(return_value=True)

    return order_manager


@pytest.fixture
def mock_signal_router():
    """Mock signal router for testing."""
    router = Mock(spec=SignalRouter)
    router.route_signals = AsyncMock(return_value=[])
    router.get_queue_size = Mock(return_value=0)
    router.clear_queue = AsyncMock()
    router.reset_state = AsyncMock()

    return router


@pytest.fixture
async def mock_timeframe_manager(mock_data_fetcher, multi_timeframe_data):
    """Mock timeframe manager for testing."""
    tf_manager = Mock(spec=TimeframeManager)
    tf_manager.data_fetcher = mock_data_fetcher
    tf_manager.add_symbol = Mock(return_value=True)
    tf_manager.fetch_multi_timeframe_data = AsyncMock(return_value=multi_timeframe_data)
    tf_manager.get_synced_data = Mock(return_value=multi_timeframe_data)
    tf_manager.initialize = AsyncMock()
    tf_manager.shutdown = AsyncMock()

    return tf_manager


@pytest.fixture
def mock_regime_detector():
    """Mock market regime detector for testing."""
    detector = Mock(spec=MarketRegimeDetector)
    detector.detect_regime = Mock(return_value="bull_market")
    detector.get_regime_confidence = Mock(return_value=0.85)
    detector.get_regime_features = Mock(return_value={
        'trend_strength': 0.7,
        'volatility': 0.15,
        'volume_trend': 0.3
    })

    return detector


@pytest.fixture
def mock_strategy_selector():
    """Mock strategy selector for testing."""
    selector = Mock(spec=StrategySelector)
    selector.enabled = True
    selector.select_strategy = Mock(return_value=None)  # Return None to use all strategies
    selector.update_performance = Mock()

    return selector


@pytest.fixture
def mock_regime_forecaster():
    """Mock regime forecaster for testing."""
    forecaster = Mock()
    forecaster.predict_regime = AsyncMock(return_value={
        'predicted_regime': 'bull_market',
        'confidence': 0.82,
        'forecast_horizon': 24,
        'features_used': ['trend_strength', 'volatility', 'volume_trend']
    })
    forecaster.get_forecast_accuracy = Mock(return_value=0.78)
    forecaster.update_model = AsyncMock()

    return forecaster


@pytest.fixture
def mock_strategy_generator():
    """Mock strategy generator for testing."""
    generator = Mock(spec=StrategyGenerator)
    generator.generate_population = Mock(return_value=[])
    generator.evolve_population = Mock(return_value=[])
    generator.evaluate_fitness = AsyncMock(return_value=[])
    generator.get_best_strategy = Mock(return_value=None)
    generator.save_population = Mock()
    generator.load_population = Mock()

    return generator


@pytest.fixture
def mock_self_healing_engine():
    """Mock self-healing engine for testing."""
    engine = Mock(spec=SelfHealingEngine)
    engine.register_component = Mock()
    engine.send_heartbeat = AsyncMock()
    engine.get_engine_stats = Mock(return_value={
        'uptime': '1:00:00',
        'total_failures_handled': 0,
        'total_recoveries_successful': 0
    })
    engine.monitoring_dashboard.get_dashboard_data = Mock(return_value={
        'system_health': {
            'overall_health': 'HEALTHY',
            'health_score': 98.5,
            'total_components': 5,
            'healthy_components': 5,
            'failing_components': 0
        }
    })

    return engine


@pytest.fixture
async def mock_bot_engine(test_config, mock_data_fetcher, mock_risk_manager,
                         mock_order_manager, mock_signal_router, mock_timeframe_manager,
                         mock_regime_detector, mock_strategy_selector,
                         mock_regime_forecaster, mock_strategy_generator,
                         mock_self_healing_engine):
    """Mock bot engine with all dependencies."""
    bot_engine = Mock(spec=BotEngine)
    bot_engine.config = test_config
    bot_engine.data_fetcher = mock_data_fetcher
    bot_engine.timeframe_manager = mock_timeframe_manager
    bot_engine.risk_manager = mock_risk_manager
    bot_engine.order_manager = mock_order_manager
    bot_engine.signal_router = mock_signal_router
    bot_engine.regime_detector = mock_regime_detector
    bot_engine.strategy_selector = mock_strategy_selector
    bot_engine.regime_forecaster = mock_regime_forecaster
    bot_engine.strategy_generator = mock_strategy_generator
    bot_engine.self_healing_engine = mock_self_healing_engine

    # Mock state
    bot_engine.state = Mock()
    bot_engine.state.running = True
    bot_engine.state.paused = False
    bot_engine.state.balance = 10000.0
    bot_engine.state.equity = 10000.0
    bot_engine.state.active_orders = 0
    bot_engine.state.open_positions = 0

    # Mock methods
    bot_engine.initialize = AsyncMock()
    bot_engine.start = AsyncMock()
    bot_engine.stop = AsyncMock()
    bot_engine.shutdown = AsyncMock()
    bot_engine._trading_cycle = AsyncMock()
    bot_engine._fetch_market_data = AsyncMock(return_value={})
    bot_engine._generate_signals = AsyncMock(return_value=[])
    bot_engine._evaluate_risk = AsyncMock(return_value=[])
    bot_engine._execute_orders = AsyncMock()

    return bot_engine


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Timer fixture for performance testing."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = datetime.now()

        def stop(self):
            self.end_time = datetime.now()

        def duration_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds() * 1000
            return 0

        def duration_seconds(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0

    return PerformanceTimer()


@pytest.fixture
def memory_monitor():
    """Memory monitoring fixture for performance testing."""
    import psutil
    import os

    class MemoryMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.initial_memory = None

        def start(self):
            self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        def get_current_memory(self):
            return self.process.memory_info().rss / 1024 / 1024  # MB

        def get_memory_delta(self):
            if self.initial_memory is not None:
                return self.get_current_memory() - self.initial_memory
            return 0

    return MemoryMonitor()


# Test data generators
@pytest.fixture
def generate_regime_data():
    """Generate test data for different market regimes."""
    def _generate_regime_data(regime_type: str, n_points: int = 500) -> pd.DataFrame:
        np.random.seed(42)

        timestamps = pd.date_range(
            start=datetime(2024, 1, 1),
            periods=n_points,
            freq='1H'
        )

        if regime_type == "bull_market":
            # Strong upward trend with moderate volatility
            base_price = 50000.0
            trend = 0.01  # Strong upward trend
            volatility = 0.015

        elif regime_type == "bear_market":
            # Strong downward trend with high volatility
            base_price = 50000.0
            trend = -0.01  # Strong downward trend
            volatility = 0.025

        elif regime_type == "sideways":
            # No trend with low volatility
            base_price = 50000.0
            trend = 0.0  # No trend
            volatility = 0.008

        elif regime_type == "high_volatility":
            # No clear trend with very high volatility
            base_price = 50000.0
            trend = 0.00005
            volatility = 0.04

        elif regime_type == "low_volatility":
            # No trend with very low volatility
            base_price = 50000.0
            trend = 0.00002
            volatility = 0.005

        else:
            raise ValueError(f"Unknown regime type: {regime_type}")

        # Generate price series
        price_changes = np.random.normal(trend, volatility/np.sqrt(24), n_points)
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Create OHLCV data
        high_mult = 1 + np.random.uniform(0, volatility*2, n_points)
        low_mult = 1 - np.random.uniform(0, volatility*2, n_points)
        close_prices = prices
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = base_price

        volumes = np.random.uniform(100, 1000, n_points)

        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': close_prices * high_mult,
            'low': close_prices * low_mult,
            'close': close_prices,
            'volume': volumes
        })

        data.set_index('timestamp', inplace=True)

        # Set regime type attribute for test compatibility
        data.attrs['regime_type'] = regime_type

        return data

    return _generate_regime_data


@pytest.fixture
def generate_strategy_population():
    """Generate test strategy population for AI strategy generator."""
    def _generate_population(size: int = 20):
        """Generate a population of test strategies."""
        from optimization.strategy_generator import StrategyGenome, StrategyGene
        from optimization.strategy_generator import StrategyComponent, IndicatorType, SignalLogic

        population = []

        for i in range(size):
            genome = StrategyGenome()

            # Add random genes
            n_genes = np.random.randint(2, 6)

            for j in range(n_genes):
                if np.random.random() < 0.6:  # 60% chance of indicator gene
                    indicator_type = np.random.choice(list(IndicatorType))
                    gene = StrategyGene(
                        component_type=StrategyComponent.INDICATOR,
                        indicator_type=indicator_type,
                        parameters={'period': np.random.randint(5, 30)}
                    )
                else:  # 40% chance of signal logic gene
                    signal_logic = np.random.choice(list(SignalLogic))
                    gene = StrategyGene(
                        component_type=StrategyComponent.SIGNAL_LOGIC,
                        signal_logic=signal_logic,
                        parameters={'threshold': np.random.uniform(0.1, 0.9)}
                    )

                genome.genes.append(gene)

            # Add fitness score
            genome.fitness = np.random.uniform(0.1, 2.0)

            population.append(genome)

        return population

    return _generate_population


# Test utilities
@pytest.fixture
def assert_performance_bounds():
    """Assert that performance is within acceptable bounds."""
    def _assert_bounds(actual: float, expected_max: float, tolerance: float = 0.1):
        """Assert that actual performance is within bounds."""
        upper_bound = expected_max * (1 + tolerance)
        assert actual <= upper_bound, f"Performance {actual} exceeds bound {upper_bound}"

    return _assert_bounds


@pytest.fixture
def assert_memory_usage():
    """Assert memory usage is within acceptable bounds."""
    def _assert_memory(usage_mb: float, max_mb: float):
        """Assert memory usage is acceptable."""
        assert usage_mb <= max_mb, f"Memory usage {usage_mb}MB exceeds limit {max_mb}MB"

    return _assert_memory


@pytest.fixture
def simulate_network_failure():
    """Simulate network failures for testing."""
    class NetworkFailureSimulator:
        def __init__(self):
            self.failure_mode = None
            self.failure_duration = 0
            self.failure_count = 0

        def set_failure_mode(self, mode: str, duration_seconds: int = 30):
            """Set network failure mode."""
            self.failure_mode = mode
            self.failure_duration = duration_seconds
            self.failure_count = 0

        def should_fail(self) -> bool:
            """Check if request should fail."""
            if self.failure_mode is None:
                return False

            if self.failure_mode == "timeout":
                return True
            elif self.failure_mode == "intermittent":
                return np.random.random() < 0.3  # 30% failure rate
            elif self.failure_mode == "temporary":
                self.failure_count += 1
                return self.failure_count <= self.failure_duration

            return False

        def reset(self):
            """Reset failure simulation."""
            self.failure_mode = None
            self.failure_duration = 0
            self.failure_count = 0

    return NetworkFailureSimulator()


# Logging configuration for tests
@pytest.fixture(autouse=True)
def configure_test_logging():
    """Configure logging for tests."""
    import logging

    # Reduce log levels for cleaner test output
    logging.getLogger('core').setLevel(logging.WARNING)
    logging.getLogger('data').setLevel(logging.WARNING)
    logging.getLogger('strategies').setLevel(logging.WARNING)
    logging.getLogger('optimization').setLevel(logging.WARNING)

    # Set test logger to INFO
    logging.getLogger('test').setLevel(logging.INFO)


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Clean up test artifacts after each test."""
    yield

    # Clean up any temporary files or resources
    # This runs after each test function
    pass


@pytest.fixture
def ensemble_config() -> Dict[str, Any]:
    """Create test configuration for ensemble manager."""
    return {
        'total_capital': 10000.0,
        'rebalance_interval_sec': 60,  # Faster for testing
        'allocation_method': 'equal_weighted',
        'min_weight': 0.1,
        'max_weight': 0.5,
        'portfolio_risk_limit': 0.1
    }
