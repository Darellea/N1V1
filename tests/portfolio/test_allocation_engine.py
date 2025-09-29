"""
Test suite for the Allocation Engine.

Tests cover various allocation methods, weight constraints, and rebalancing logic.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict

import pytest

from portfolio.allocation_engine import AllocationEngine, create_allocation_engine


class TestAllocationEngine:
    """Test the allocation engine functionality."""

    @pytest.fixture
    def allocation_config(self) -> Dict[str, Any]:
        """Create test configuration for allocation engine."""
        return {
            "min_weight": 0.05,
            "max_weight": 0.4,
            "risk_free_rate": 0.02,
            "performance_window_days": 30,
            "rebalance_threshold": 0.05,
        }

    @pytest.fixture
    def allocation_engine(self, allocation_config: Dict[str, Any]) -> AllocationEngine:
        """Create allocation engine for testing."""
        return AllocationEngine(allocation_config)

    def test_initialization(
        self, allocation_engine: AllocationEngine, allocation_config: Dict[str, Any]
    ):
        """Test allocation engine initialization."""
        assert allocation_engine.min_weight == 0.05
        assert allocation_engine.max_weight == 0.4
        assert allocation_engine.risk_free_rate == 0.02
        assert allocation_engine.performance_window_days == 30
        assert allocation_engine.rebalance_threshold == 0.05

    def test_calculate_equal_weights(self, allocation_engine: AllocationEngine):
        """Test equal weight allocation calculation."""
        strategy_performance = {
            "strategy1": [{"daily_return": 0.01, "pnl": 100.0, "trades": 5}],
            "strategy2": [{"daily_return": 0.02, "pnl": 200.0, "trades": 5}],
            "strategy3": [{"daily_return": 0.015, "pnl": 150.0, "trades": 5}],
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "equal_weighted", Decimal("10000")
        )

        # All strategies should get equal weight
        assert len(allocations) == 3
        for strategy_id, allocation in allocations.items():
            assert allocation["weight"] == pytest.approx(1.0 / 3.0, abs=0.01)
            assert allocation["capital_allocated"] == Decimal("10000") * Decimal(
                str(allocation["weight"])
            )

    def test_calculate_sharpe_weights(self, allocation_engine: AllocationEngine):
        """Test Sharpe-weighted allocation calculation."""
        # Create performance data with different Sharpe ratios
        base_time = datetime.now()

        strategy_performance = {
            "high_sharpe": [
                {
                    "daily_return": 0.05,
                    "pnl": 500.0,
                    "trades": 10,
                    "timestamp": base_time,
                },
                {
                    "daily_return": 0.04,
                    "pnl": 400.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=1),
                },
                {
                    "daily_return": 0.06,
                    "pnl": 600.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=2),
                },
                {
                    "daily_return": 0.03,
                    "pnl": 300.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=3),
                },
                {
                    "daily_return": 0.05,
                    "pnl": 500.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=4),
                },
            ],
            "low_sharpe": [
                {
                    "daily_return": 0.01,
                    "pnl": 100.0,
                    "trades": 10,
                    "timestamp": base_time,
                },
                {
                    "daily_return": 0.015,
                    "pnl": 150.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=1),
                },
                {
                    "daily_return": 0.008,
                    "pnl": 80.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=2),
                },
                {
                    "daily_return": 0.012,
                    "pnl": 120.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=3),
                },
                {
                    "daily_return": 0.009,
                    "pnl": 90.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=4),
                },
            ],
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "sharpe_weighted", Decimal("10000")
        )

        # High Sharpe strategy should get more weight
        assert (
            allocations["high_sharpe"]["weight"] > allocations["low_sharpe"]["weight"]
        )
        assert (
            abs(
                allocations["high_sharpe"]["weight"]
                + allocations["low_sharpe"]["weight"]
                - 1.0
            )
            < 0.01
        )

    def test_weight_constraints(self, allocation_engine: AllocationEngine):
        """Test weight constraints application."""
        # Create performance data that would result in extreme weights
        strategy_performance = {
            "excellent": [
                {"daily_return": 0.10, "pnl": 1000.0, "trades": 10}
            ],  # Very high return
            "poor": [
                {"daily_return": -0.05, "pnl": -500.0, "trades": 10}
            ],  # Very low return
            "mediocre": [
                {"daily_return": 0.01, "pnl": 100.0, "trades": 10}
            ],  # Normal return
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "sharpe_weighted", Decimal("10000")
        )

        # Check that all weights are within constraints
        for strategy_id, allocation in allocations.items():
            assert (
                allocation_engine.min_weight
                <= allocation["weight"]
                <= allocation_engine.max_weight
            )

        # Total weight should still sum to 1.0
        total_weight = sum(allocation["weight"] for allocation in allocations.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_insufficient_performance_data(self, allocation_engine: AllocationEngine):
        """Test handling of insufficient performance data."""
        # Strategy with very little performance data
        strategy_performance = {
            "new_strategy": [
                {"daily_return": 0.01, "pnl": 10.0, "trades": 1}
            ]  # Only 1 data point
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "sharpe_weighted", Decimal("10000")
        )

        # Should still allocate weight (fallback to equal weighting)
        assert len(allocations) == 1
        assert allocations["new_strategy"]["weight"] == 1.0
        assert allocations["new_strategy"]["capital_allocated"] == Decimal("10000")

    def test_empty_strategy_performance(self, allocation_engine: AllocationEngine):
        """Test handling of empty strategy performance."""
        allocations = allocation_engine.calculate_allocations(
            {}, "equal_weighted", Decimal("10000")
        )

        assert len(allocations) == 0

    def test_should_rebalance_positive(self, allocation_engine: AllocationEngine):
        """Test rebalance trigger with significant allocation changes."""
        current_allocations = {
            "strategy1": {"weight": 0.5},
            "strategy2": {"weight": 0.5},
        }

        new_allocations = {
            "strategy1": {"weight": 0.7},  # +20% change
            "strategy2": {"weight": 0.3},  # -20% change
        }

        should_rebalance = allocation_engine.should_rebalance(
            current_allocations, new_allocations
        )

        assert should_rebalance is True

    def test_should_rebalance_negative(self, allocation_engine: AllocationEngine):
        """Test no rebalance trigger with small allocation changes."""
        current_allocations = {
            "strategy1": {"weight": 0.5},
            "strategy2": {"weight": 0.5},
        }

        new_allocations = {
            "strategy1": {"weight": 0.52},  # +2% change (below threshold)
            "strategy2": {"weight": 0.48},  # -2% change (below threshold)
        }

        should_rebalance = allocation_engine.should_rebalance(
            current_allocations, new_allocations
        )

        assert should_rebalance is False

    def test_should_rebalance_missing_strategy(
        self, allocation_engine: AllocationEngine
    ):
        """Test rebalance trigger when strategies are missing."""
        current_allocations = {
            "strategy1": {"weight": 0.5},
            "strategy2": {"weight": 0.5},
        }

        new_allocations = {
            "strategy1": {"weight": 0.5}
            # strategy2 is missing
        }

        should_rebalance = allocation_engine.should_rebalance(
            current_allocations, new_allocations
        )

        assert should_rebalance is True

    def test_sortino_weighted_allocation(self, allocation_engine: AllocationEngine):
        """Test Sortino-weighted allocation calculation."""
        base_time = datetime.now()

        strategy_performance = {
            "low_downside": [
                {
                    "daily_return": 0.03,
                    "pnl": 300.0,
                    "trades": 10,
                    "timestamp": base_time,
                },
                {
                    "daily_return": 0.04,
                    "pnl": 400.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=1),
                },
                {
                    "daily_return": 0.02,
                    "pnl": 200.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=2),
                },
                {
                    "daily_return": 0.035,
                    "pnl": 350.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=3),
                },
                {
                    "daily_return": 0.025,
                    "pnl": 250.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=4),
                },
            ],
            "high_downside": [
                {
                    "daily_return": 0.02,
                    "pnl": 200.0,
                    "trades": 10,
                    "timestamp": base_time,
                },
                {
                    "daily_return": -0.03,
                    "pnl": -300.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=1),
                },  # Loss
                {
                    "daily_return": 0.04,
                    "pnl": 400.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=2),
                },
                {
                    "daily_return": -0.02,
                    "pnl": -200.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=3),
                },  # Loss
                {
                    "daily_return": 0.03,
                    "pnl": 300.0,
                    "trades": 10,
                    "timestamp": base_time + timedelta(days=4),
                },
            ],
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "sortino_weighted", Decimal("10000")
        )

        # Strategy with lower downside risk should get more weight
        assert (
            allocations["low_downside"]["weight"]
            > allocations["high_downside"]["weight"]
        )

    def test_kelly_weighted_allocation(self, allocation_engine: AllocationEngine):
        """Test Kelly-weighted allocation calculation."""
        base_time = datetime.now()

        strategy_performance = {
            "good_kelly": [
                {"pnl": 100.0, "timestamp": base_time},
                {"pnl": 150.0, "timestamp": base_time + timedelta(days=1)},
                {"pnl": 120.0, "timestamp": base_time + timedelta(days=2)},
                {"pnl": 180.0, "timestamp": base_time + timedelta(days=3)},
                {"pnl": 140.0, "timestamp": base_time + timedelta(days=4)},
                {"pnl": 160.0, "timestamp": base_time + timedelta(days=5)},
                {"pnl": 130.0, "timestamp": base_time + timedelta(days=6)},
                {"pnl": 170.0, "timestamp": base_time + timedelta(days=7)},
                {"pnl": 110.0, "timestamp": base_time + timedelta(days=8)},
                {"pnl": 190.0, "timestamp": base_time + timedelta(days=9)},
            ],
            "poor_kelly": [
                {"pnl": -100.0, "timestamp": base_time},
                {"pnl": 50.0, "timestamp": base_time + timedelta(days=1)},
                {"pnl": -80.0, "timestamp": base_time + timedelta(days=2)},
                {"pnl": 30.0, "timestamp": base_time + timedelta(days=3)},
                {"pnl": -60.0, "timestamp": base_time + timedelta(days=4)},
                {"pnl": 40.0, "timestamp": base_time + timedelta(days=5)},
                {"pnl": -70.0, "timestamp": base_time + timedelta(days=6)},
                {"pnl": 20.0, "timestamp": base_time + timedelta(days=7)},
                {"pnl": -50.0, "timestamp": base_time + timedelta(days=8)},
                {"pnl": 60.0, "timestamp": base_time + timedelta(days=9)},
            ],
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "kelly_weighted", Decimal("10000")
        )

        # Strategy with better risk-adjusted returns should get more weight
        assert allocations["good_kelly"]["weight"] > allocations["poor_kelly"]["weight"]

    def test_volatility_targeted_allocation(self, allocation_engine: AllocationEngine):
        """Test volatility-targeted allocation calculation."""
        base_time = datetime.now()

        strategy_performance = {
            "low_vol": [
                {"daily_return": 0.01, "timestamp": base_time},
                {"daily_return": 0.012, "timestamp": base_time + timedelta(days=1)},
                {"daily_return": 0.008, "timestamp": base_time + timedelta(days=2)},
                {"daily_return": 0.011, "timestamp": base_time + timedelta(days=3)},
                {"daily_return": 0.009, "timestamp": base_time + timedelta(days=4)},
            ],
            "high_vol": [
                {"daily_return": 0.05, "timestamp": base_time},
                {"daily_return": -0.03, "timestamp": base_time + timedelta(days=1)},
                {"daily_return": 0.08, "timestamp": base_time + timedelta(days=2)},
                {"daily_return": -0.06, "timestamp": base_time + timedelta(days=3)},
                {"daily_return": 0.04, "timestamp": base_time + timedelta(days=4)},
            ],
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "volatility_targeted", Decimal("10000")
        )

        # Low volatility strategy should get more weight (risk parity)
        assert allocations["low_vol"]["weight"] > allocations["high_vol"]["weight"]


class TestGlobalFunctions:
    """Test global allocation engine functions."""

    def test_create_allocation_engine(self, allocation_config: Dict[str, Any]):
        """Test creating an allocation engine."""
        engine = create_allocation_engine(allocation_config)

        assert isinstance(engine, AllocationEngine)
        assert engine.min_weight == 0.05
        assert engine.max_weight == 0.4


class TestEdgeCases:
    """Test edge cases in allocation engine."""

    @pytest.fixture
    def allocation_engine(self) -> AllocationEngine:
        """Create allocation engine for testing."""
        return AllocationEngine()

    def test_single_strategy(self, allocation_engine: AllocationEngine):
        """Test allocation with single strategy."""
        strategy_performance = {
            "only_strategy": [{"daily_return": 0.02, "pnl": 200.0, "trades": 5}]
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "equal_weighted", Decimal("10000")
        )

        assert len(allocations) == 1
        assert allocations["only_strategy"]["weight"] == 1.0
        assert allocations["only_strategy"]["capital_allocated"] == Decimal("10000")

    def test_zero_returns(self, allocation_engine: AllocationEngine):
        """Test allocation with zero returns."""
        strategy_performance = {
            "flat_strategy": [
                {"daily_return": 0.0, "pnl": 0.0, "trades": 5},
                {"daily_return": 0.0, "pnl": 0.0, "trades": 5},
                {"daily_return": 0.0, "pnl": 0.0, "trades": 5},
            ]
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "sharpe_weighted", Decimal("10000")
        )

        # Should still allocate weight
        assert len(allocations) == 1
        assert allocations["flat_strategy"]["weight"] == 1.0

    def test_negative_returns(self, allocation_engine: AllocationEngine):
        """Test allocation with negative returns."""
        strategy_performance = {
            "losing_strategy": [
                {"daily_return": -0.02, "pnl": -200.0, "trades": 5},
                {"daily_return": -0.015, "pnl": -150.0, "trades": 5},
                {"daily_return": -0.025, "pnl": -250.0, "trades": 5},
            ]
        }

        allocations = allocation_engine.calculate_allocations(
            strategy_performance, "sharpe_weighted", Decimal("10000")
        )

        # Should still allocate minimum weight
        assert len(allocations) == 1
        assert allocations["losing_strategy"]["weight"] >= allocation_engine.min_weight


if __name__ == "__main__":
    pytest.main([__file__])
