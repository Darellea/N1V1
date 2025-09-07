"""
Test suite for the Strategy Ensemble Manager.

Tests cover strategy management, signal routing, allocation updates, and ensemble coordination.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from portfolio.strategy_ensemble import (
    StrategyEnsembleManager,
    StrategyAllocation,
    EnsemblePerformance,
    EnsembleSignal,
    create_ensemble_manager
)
from core.contracts import TradingSignal
from core.signal_router.events import EventType


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        self.signals_generated = []

    async def generate_signal(self, market_data: Dict[str, Any]) -> TradingSignal:
        """Generate a mock trading signal."""
        signal = TradingSignal(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now(),
            strategy_id=self.strategy_id
        )
        self.signals_generated.append(signal)
        return signal


class TestStrategyEnsembleManager:
    """Test the main strategy ensemble manager."""

    @pytest.fixture
    def ensemble_config(self) -> Dict[str, Any]:
        """Create test configuration for ensemble manager."""
        return {
            'total_capital': 10000.0,
            'rebalance_interval_sec': 60,  # Faster for testing
            'allocation_method': 'equal_weighted',
            'min_weight': 0.1,
            'max_weight': 0.5,
            'portfolio_risk_limit': 0.1
        }

    @pytest.fixture
    def ensemble_manager(self, ensemble_config: Dict[str, Any]) -> StrategyEnsembleManager:
        """Create ensemble manager for testing."""
        return StrategyEnsembleManager(ensemble_config)

    def test_initialization(self, ensemble_manager: StrategyEnsembleManager, ensemble_config: Dict[str, Any]):
        """Test ensemble manager initialization."""
        assert ensemble_manager.total_capital == Decimal('10000.0')
        assert ensemble_manager.rebalance_interval_sec == 60
        assert ensemble_manager.allocation_method == 'equal_weighted'
        assert ensemble_manager.min_weight == 0.1
        assert ensemble_manager.max_weight == 0.5
        assert len(ensemble_manager.strategies) == 0
        assert len(ensemble_manager.allocations) == 0
        assert not ensemble_manager._running

    @pytest.mark.asyncio
    async def test_add_strategy(self, ensemble_manager: StrategyEnsembleManager):
        """Test adding a strategy to the ensemble."""
        strategy = MockStrategy("test_strategy")

        await ensemble_manager.add_strategy("test_strategy", strategy, 0.5)

        assert "test_strategy" in ensemble_manager.strategies
        assert "test_strategy" in ensemble_manager.allocations
        assert ensemble_manager.allocations["test_strategy"].weight == 0.5
        assert ensemble_manager.allocations["test_strategy"].capital_allocated == Decimal('5000.0')

    @pytest.mark.asyncio
    async def test_add_duplicate_strategy(self, ensemble_manager: StrategyEnsembleManager):
        """Test adding a duplicate strategy."""
        strategy1 = MockStrategy("test_strategy")
        strategy2 = MockStrategy("test_strategy")

        await ensemble_manager.add_strategy("test_strategy", strategy1)
        await ensemble_manager.add_strategy("test_strategy", strategy2)  # Should not add

        assert len(ensemble_manager.strategies) == 1
        assert ensemble_manager.strategies["test_strategy"] is strategy1

    @pytest.mark.asyncio
    async def test_remove_strategy(self, ensemble_manager: StrategyEnsembleManager):
        """Test removing a strategy from the ensemble."""
        strategy = MockStrategy("test_strategy")
        await ensemble_manager.add_strategy("test_strategy", strategy)

        await ensemble_manager.remove_strategy("test_strategy")

        assert "test_strategy" not in ensemble_manager.strategies
        assert "test_strategy" not in ensemble_manager.allocations

    @pytest.mark.asyncio
    async def test_remove_nonexistent_strategy(self, ensemble_manager: StrategyEnsembleManager):
        """Test removing a strategy that doesn't exist."""
        # Should not raise an exception
        await ensemble_manager.remove_strategy("nonexistent")

        assert len(ensemble_manager.strategies) == 0

    @pytest.mark.asyncio
    async def test_route_signal_success(self, ensemble_manager: StrategyEnsembleManager):
        """Test successful signal routing."""
        strategy = MockStrategy("test_strategy")
        await ensemble_manager.add_strategy("test_strategy", strategy)

        signal = TradingSignal(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now(),
            strategy_id="test_strategy"
        )

        ensemble_signal = await ensemble_manager.route_signal(signal, "test_strategy")

        assert ensemble_signal is not None
        assert ensemble_signal.original_signal == signal
        assert ensemble_signal.strategy_id == "test_strategy"
        assert ensemble_signal.allocated_weight == 1.0  # Single strategy gets full weight
        assert ensemble_signal.allocated_quantity == Decimal('1.0')

    @pytest.mark.asyncio
    async def test_route_signal_unknown_strategy(self, ensemble_manager: StrategyEnsembleManager):
        """Test routing signal from unknown strategy."""
        signal = TradingSignal(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now(),
            strategy_id="unknown"
        )

        ensemble_signal = await ensemble_manager.route_signal(signal, "unknown")

        assert ensemble_signal is None

    @pytest.mark.asyncio
    async def test_route_signal_risk_limit_exceeded(self, ensemble_manager: StrategyEnsembleManager):
        """Test signal rejection due to risk limits."""
        strategy = MockStrategy("test_strategy")
        await ensemble_manager.add_strategy("test_strategy", strategy)

        # Create a very large signal that exceeds risk limits
        signal = TradingSignal(
            symbol="BTC/USDT",
            side="buy",
            quantity=1000.0,  # Very large quantity
            price=50000.0,
            timestamp=datetime.now(),
            strategy_id="test_strategy"
        )

        ensemble_signal = await ensemble_manager.route_signal(signal, "test_strategy")

        # Should be rejected due to risk limits
        assert ensemble_signal is None

    @pytest.mark.asyncio
    async def test_update_performance(self, ensemble_manager: StrategyEnsembleManager):
        """Test updating strategy performance."""
        strategy = MockStrategy("test_strategy")
        await ensemble_manager.add_strategy("test_strategy", strategy)

        performance_data = {
            'daily_return': 0.02,
            'pnl': 200.0,
            'trades': 5,
            'timestamp': datetime.now()
        }

        await ensemble_manager.update_performance("test_strategy", performance_data)

        assert len(ensemble_manager.performance_history["test_strategy"]) == 1
        assert ensemble_manager.performance_history["test_strategy"][0] == performance_data

    def test_get_strategy_allocations(self, ensemble_manager: StrategyEnsembleManager):
        """Test getting strategy allocations."""
        allocations = ensemble_manager.get_strategy_allocations()

        assert isinstance(allocations, dict)
        assert len(allocations) == 0  # No strategies added yet

    def test_get_ensemble_status(self, ensemble_manager: StrategyEnsembleManager):
        """Test getting ensemble status."""
        status = ensemble_manager.get_ensemble_status()

        required_keys = [
            'total_capital', 'active_strategies', 'total_allocations',
            'portfolio_exposure', 'allocation_method', 'running'
        ]

        for key in required_keys:
            assert key in status

        assert status['total_capital'] == 10000.0
        assert status['active_strategies'] == 0
        assert not status['running']

    @pytest.mark.asyncio
    async def test_get_portfolio_performance(self, ensemble_manager: StrategyEnsembleManager):
        """Test getting portfolio performance."""
        performance = await ensemble_manager.get_portfolio_performance()

        assert isinstance(performance, EnsemblePerformance)
        assert performance.total_capital == Decimal('10000.0')
        assert performance.total_pnl == Decimal('0')

    @pytest.mark.asyncio
    async def test_start_stop_ensemble(self, ensemble_manager: StrategyEnsembleManager):
        """Test starting and stopping the ensemble."""
        assert not ensemble_manager._running

        await ensemble_manager.start()
        assert ensemble_manager._running

        await ensemble_manager.stop()
        assert not ensemble_manager._running

    @pytest.mark.asyncio
    async def test_multiple_strategies_equal_weight(self, ensemble_manager: StrategyEnsembleManager):
        """Test multiple strategies with equal weight allocation."""
        strategy1 = MockStrategy("strategy1")
        strategy2 = MockStrategy("strategy2")

        await ensemble_manager.add_strategy("strategy1", strategy1)
        await ensemble_manager.add_strategy("strategy2", strategy2)

        # Both should get equal weight
        assert ensemble_manager.allocations["strategy1"].weight == 0.5
        assert ensemble_manager.allocations["strategy2"].weight == 0.5

    @pytest.mark.asyncio
    async def test_weight_constraints(self, ensemble_manager: StrategyEnsembleManager):
        """Test weight constraints application."""
        # Add a strategy with weight below minimum
        strategy = MockStrategy("test_strategy")
        await ensemble_manager.add_strategy("test_strategy", strategy, 0.01)  # Below min_weight

        # Should be adjusted to minimum
        assert ensemble_manager.allocations["test_strategy"].weight >= ensemble_manager.min_weight


class TestEnsembleSignal:
    """Test ensemble signal data structure."""

    def test_ensemble_signal_creation(self):
        """Test creating an ensemble signal."""
        original_signal = TradingSignal(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now(),
            strategy_id="test_strategy"
        )

        ensemble_signal = EnsembleSignal(
            original_signal=original_signal,
            strategy_id="test_strategy",
            allocated_weight=0.5,
            allocated_quantity=Decimal('0.5'),
            ensemble_context={
                "allocation_weight": 0.5,
                "capital_allocated": 5000.0,
                "portfolio_exposure": 25000.0,
                "strategy_performance": 0.8
            }
        )

        assert ensemble_signal.original_signal == original_signal
        assert ensemble_signal.strategy_id == "test_strategy"
        assert ensemble_signal.allocated_weight == 0.5
        assert ensemble_signal.allocated_quantity == Decimal('0.5')
        assert ensemble_signal.ensemble_context["allocation_weight"] == 0.5


class TestStrategyAllocation:
    """Test strategy allocation data structure."""

    def test_strategy_allocation_creation(self):
        """Test creating a strategy allocation."""
        allocation = StrategyAllocation(
            strategy_id="test_strategy",
            weight=0.3,
            capital_allocated=Decimal('3000.0'),
            performance_score=0.8
        )

        assert allocation.strategy_id == "test_strategy"
        assert allocation.weight == 0.3
        assert allocation.capital_allocated == Decimal('3000.0')
        assert allocation.performance_score == 0.8
        assert isinstance(allocation.last_updated, datetime)


class TestEnsemblePerformance:
    """Test ensemble performance data structure."""

    def test_ensemble_performance_creation(self):
        """Test creating ensemble performance metrics."""
        performance = EnsemblePerformance(
            total_capital=Decimal('10000.0'),
            total_pnl=Decimal('500.0'),
            total_trades=25,
            win_rate=0.6,
            sharpe_ratio=1.5
        )

        assert performance.total_capital == Decimal('10000.0')
        assert performance.total_pnl == Decimal('500.0')
        assert performance.total_trades == 25
        assert performance.win_rate == 0.6
        assert performance.sharpe_ratio == 1.5


class TestGlobalFunctions:
    """Test global ensemble functions."""

    def test_create_ensemble_manager(self, ensemble_config: Dict[str, Any]):
        """Test creating an ensemble manager."""
        manager = create_ensemble_manager(ensemble_config)

        assert isinstance(manager, StrategyEnsembleManager)
        assert manager.total_capital == Decimal('10000.0')


class TestIntegrationScenarios:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_ensemble_workflow(self, ensemble_config: Dict[str, Any]):
        """Test complete ensemble workflow."""
        manager = StrategyEnsembleManager(ensemble_config)

        # Add strategies
        strategy1 = MockStrategy("rsi_strategy")
        strategy2 = MockStrategy("ema_strategy")

        await manager.add_strategy("rsi_strategy", strategy1, 0.6)
        await manager.add_strategy("ema_strategy", strategy2, 0.4)

        # Start ensemble
        await manager.start()

        # Route signals
        signal1 = TradingSignal(
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now(),
            strategy_id="rsi_strategy"
        )

        ensemble_signal = await manager.route_signal(signal1, "rsi_strategy")
        assert ensemble_signal is not None
        assert ensemble_signal.allocated_weight == 0.6

        # Update performance
        await manager.update_performance("rsi_strategy", {
            'daily_return': 0.02,
            'pnl': 100.0,
            'trades': 2,
            'timestamp': datetime.now()
        })

        # Get portfolio performance
        performance = await manager.get_portfolio_performance()
        assert performance.total_pnl == Decimal('100.0')

        # Stop ensemble
        await manager.stop()

    @pytest.mark.asyncio
    async def test_rebalancing_workflow(self, ensemble_config: Dict[str, Any]):
        """Test rebalancing workflow."""
        # Use sharpe-weighted allocation for rebalancing test
        ensemble_config['allocation_method'] = 'sharpe_weighted'
        ensemble_config['rebalance_interval_sec'] = 1  # Very fast for testing

        manager = StrategyEnsembleManager(ensemble_config)

        # Add strategies
        strategy1 = MockStrategy("good_strategy")
        strategy2 = MockStrategy("poor_strategy")

        await manager.add_strategy("good_strategy", strategy1)
        await manager.add_strategy("poor_strategy", strategy2)

        # Add performance data to trigger rebalancing
        await manager.update_performance("good_strategy", {
            'daily_return': 0.05,  # Good performance
            'pnl': 500.0,
            'trades': 10,
            'timestamp': datetime.now()
        })

        await manager.update_performance("poor_strategy", {
            'daily_return': -0.02,  # Poor performance
            'pnl': -200.0,
            'trades': 10,
            'timestamp': datetime.now()
        })

        # Initial weights should be equal
        assert abs(manager.allocations["good_strategy"].weight - 0.5) < 0.01
        assert abs(manager.allocations["poor_strategy"].weight - 0.5) < 0.01

        # Note: In a real scenario, rebalancing would happen automatically
        # but for testing we can manually trigger it
        await manager._rebalance_allocations()

        # Good strategy should get more weight after rebalancing
        assert manager.allocations["good_strategy"].weight > manager.allocations["poor_strategy"].weight


if __name__ == "__main__":
    pytest.main([__file__])
