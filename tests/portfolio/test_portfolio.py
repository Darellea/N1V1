"""
Unit tests for portfolio management module.
"""

from decimal import Decimal
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from portfolio import (
    EqualWeightAllocator,
    MomentumWeightAllocator,
    PortfolioHedger,
    PortfolioManager,
    RiskParityAllocator,
)
from portfolio.portfolio_manager import Position


class TestPortfolioManager:
    """Test PortfolioManager class."""

    def test_initialization(self):
        """Test PortfolioManager initialization."""
        config = {
            "rotation": {"method": "momentum", "top_n": 5},
            "rebalancing": {"mode": "threshold", "scheme": "equal_weight"},
            "hedging": {"enabled": False},
        }

        pm = PortfolioManager(config, initial_balance=Decimal("10000"))

        assert pm.cash_balance == Decimal("10000")
        assert len(pm.positions) == 0
        assert pm.config == config

    def test_update_prices(self):
        """Test price updates."""
        config = {}
        pm = PortfolioManager(config)

        # Add a mock position
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
        )
        pm.positions["BTC/USDT"] = position

        # Update prices
        price_data = {"BTC/USDT": Decimal("51000")}
        pm.update_prices(price_data)

        # Check that position P&L was updated
        assert pm.positions["BTC/USDT"].current_price == Decimal("51000")
        assert pm.positions["BTC/USDT"].unrealized_pnl == Decimal(
            "1000"
        )  # 1 * (51000 - 50000)

    def test_portfolio_value_calculation(self):
        """Test portfolio value calculation."""
        config = {}
        pm = PortfolioManager(config, initial_balance=Decimal("10000"))

        # Add positions
        pm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
        )
        pm.positions["ETH/USDT"] = Position(
            symbol="ETH/USDT",
            quantity=Decimal("10"),
            entry_price=Decimal("3000"),
            current_price=Decimal("3100"),
        )

        portfolio_value = pm.get_portfolio_value()
        expected_value = (
            Decimal("10000") + Decimal("51000") + Decimal("31000")
        )  # cash + BTC + ETH
        assert portfolio_value == expected_value

    def test_portfolio_metrics(self):
        """Test portfolio metrics calculation."""
        config = {}
        pm = PortfolioManager(config, initial_balance=Decimal("10000"))

        # Add positions with P&L
        pm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
        )

        metrics = pm.get_portfolio_metrics()

        assert metrics.total_value > Decimal("10000")
        assert metrics.total_pnl > 0
        assert metrics.num_positions == 1
        assert metrics.num_assets == 1

    def test_asset_rotation_momentum(self):
        """Test momentum-based asset rotation."""
        config = {"rotation": {"method": "momentum", "lookback_days": 30, "top_n": 2}}
        pm = PortfolioManager(config)

        # Create mock market data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "BTC/USDT": np.random.randn(50).cumsum() + 50000,
                "ETH/USDT": np.random.randn(50).cumsum() + 3000,
                "ADA/USDT": np.random.randn(50).cumsum() + 1,
            },
            index=dates,
        )

        # Mock strategy signals
        strategy_signals = {
            "BTC/USDT": [{"signal_strength": 0.8}],
            "ETH/USDT": [{"signal_strength": 0.6}],
            "ADA/USDT": [{"signal_strength": 0.4}],
        }

        selected_assets = pm.rotate_assets(strategy_signals, data)

        assert len(selected_assets) <= 2  # Should select top 2
        assert all(
            asset in ["BTC/USDT", "ETH/USDT", "ADA/USDT"] for asset in selected_assets
        )

    def test_rebalancing_threshold(self):
        """Test threshold-based rebalancing."""
        config = {
            "rebalancing": {
                "mode": "threshold",
                "threshold": 0.1,
                "scheme": "equal_weight",
            }
        }
        pm = PortfolioManager(config, initial_balance=Decimal("10000"))

        # Add positions
        pm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            quantity=Decimal("1"),
            entry_price=Decimal("50000"),
            current_price=Decimal("50000"),
        )

        # Current allocation is ~83% BTC, 17% cash
        # Target equal allocation should trigger rebalancing
        target_allocations = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}

        result = pm.rebalance(target_allocations)

        assert result["rebalanced"] == True
        assert len(result["trades"]) > 0

    def test_rebalancing_no_change(self):
        """Test rebalancing when no change is needed."""
        config = {
            "rebalancing": {
                "mode": "threshold",
                "threshold": 0.1,
                "scheme": "equal_weight",
            }
        }
        pm = PortfolioManager(config, initial_balance=Decimal("10000"))

        # No positions, should not rebalance
        target_allocations = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}

        result = pm.rebalance(target_allocations)

        assert result["rebalanced"] == False
        assert result["reason"] == "within_threshold"


class TestCapitalAllocators:
    """Test capital allocation strategies."""

    def test_equal_weight_allocation(self):
        """Test equal weight allocation."""
        allocator = EqualWeightAllocator()

        assets = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        allocations = allocator.allocate(assets)

        assert len(allocations) == 3
        assert all(
            abs(allocation - 1 / 3) < 0.001 for allocation in allocations.values()
        )
        assert allocator.validate_allocations(allocations)

    def test_risk_parity_allocation(self):
        """Test risk parity allocation."""
        config = {"lookback_period": 30}
        allocator = RiskParityAllocator(config)

        assets = ["BTC/USDT", "ETH/USDT"]

        # Create mock data with different volatilities
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "BTC/USDT": np.random.randn(50) * 0.02 + 50000,  # Low volatility
                "ETH/USDT": np.random.randn(50) * 0.05 + 3000,  # High volatility
            },
            index=dates,
        )

        allocations = allocator.allocate(assets, data)

        assert len(allocations) == 2
        assert allocator.validate_allocations(allocations)
        # Higher volatility asset should get lower allocation
        assert allocations["ETH/USDT"] < allocations["BTC/USDT"]

    def test_momentum_weighted_allocation(self):
        """Test momentum weighted allocation."""
        config = {"lookback_period": 30}
        allocator = MomentumWeightAllocator(config)

        assets = ["BTC/USDT", "ETH/USDT"]

        # Create mock data with different momentum
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "BTC/USDT": np.linspace(50000, 51000, 50),  # Strong uptrend
                "ETH/USDT": np.linspace(3000, 2950, 50),  # Downtrend
            },
            index=dates,
        )

        allocations = allocator.allocate(assets, data)

        assert len(allocations) == 2
        assert allocator.validate_allocations(allocations)
        # Stronger momentum asset should get higher allocation
        assert allocations["BTC/USDT"] > allocations["ETH/USDT"]


class TestPortfolioHedger:
    """Test PortfolioHedger class."""

    def test_hedge_evaluation_no_trigger(self):
        """Test hedging when no trigger conditions are met."""
        config = {
            "enabled": True,
            "max_stablecoin_pct": 0.3,
            "trigger": {"adx_below": 15},
        }
        hedger = PortfolioHedger(config)

        positions = {"BTC/USDT": Mock()}
        market_conditions = {"adx": 25}  # Above threshold

        result = hedger.evaluate_hedging(positions, market_conditions)

        assert result is None  # No hedging should be triggered

    def test_hedge_evaluation_triggered(self):
        """Test hedging when trigger conditions are met."""
        config = {
            "enabled": True,
            "max_stablecoin_pct": 0.3,
            "trigger": {"adx_below": 15},
        }
        hedger = PortfolioHedger(config)

        positions = {"BTC/USDT": Mock()}
        market_conditions = {"adx": 10}  # Below threshold

        result = hedger.evaluate_hedging(positions, market_conditions)

        assert result is not None
        assert "action_type" in result
        assert "trades" in result

    def test_stablecoin_rotation_hedge(self):
        """Test stablecoin rotation hedging strategy."""
        config = {"enabled": True, "max_stablecoin_pct": 0.3}
        hedger = PortfolioHedger(config)

        # Mock positions
        positions = {
            "BTC/USDT": Mock(market_value=Decimal("7000")),
            "ETH/USDT": Mock(market_value=Decimal("3000")),
        }

        trades = hedger._stablecoin_rotation_hedge(positions)

        # Should generate sell trades to buy stablecoins
        assert len(trades) > 0
        assert all(trade["side"] == "sell" for trade in trades)
        assert all(
            "hedge_stablecoin_rotation" in trade.get("reason", "") for trade in trades
        )


class TestPosition:
    """Test Position dataclass."""

    def test_position_initialization(self):
        """Test Position initialization."""
        position = Position(
            symbol="BTC/USDT", quantity=Decimal("1"), entry_price=Decimal("50000")
        )

        assert position.symbol == "BTC/USDT"
        assert position.quantity == Decimal("1")
        assert position.entry_price == Decimal("50000")
        assert position.unrealized_pnl == Decimal("0")

    def test_position_pnl_calculation(self):
        """Test P&L calculation."""
        position = Position(
            symbol="BTC/USDT", quantity=Decimal("1"), entry_price=Decimal("50000")
        )

        # Update price
        position.update_pnl(Decimal("51000"))

        assert position.current_price == Decimal("51000")
        assert position.unrealized_pnl == Decimal("1000")  # 1 * (51000 - 50000)

    def test_position_market_value(self):
        """Test market value calculation."""
        position = Position(
            symbol="BTC/USDT",
            quantity=Decimal("2"),
            entry_price=Decimal("50000"),
            current_price=Decimal("51000"),
        )

        assert position.market_value == Decimal("102000")  # 2 * 51000


if __name__ == "__main__":
    pytest.main([__file__])
