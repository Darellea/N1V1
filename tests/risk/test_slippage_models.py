"""
Comprehensive test suite for dynamic slippage models in risk management.

Tests cover:
- Slippage model validation (constant, linear, square root)
- Liquidity scenario testing
- Accuracy benchmarks
- Performance testing with timeouts
- Integration with position sizing
- Edge cases (illiquid markets, large orders)
"""

import time
from decimal import Decimal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.contracts import SignalStrength, SignalType, TradingSignal
from risk.risk_manager import RiskManager


@pytest.fixture
def slippage_config():
    """Fixture providing slippage model configuration."""
    return {
        "slippage_models": {
            "constant": {
                "base_slippage": 0.001,  # 0.1%
            },
            "linear": {
                "base_slippage": 0.0005,  # 0.05%
                "order_size_factor": 0.0001,  # Additional slippage per unit of order size
            },
            "square_root": {
                "base_slippage": 0.0002,  # 0.02%
                "volatility_factor": 0.5,  # Multiplier for volatility impact
            },
        },
        "liquidity_thresholds": {
            "high_liquidity": {"volume_threshold": 5000000, "spread_threshold": 0.0006},
            "medium_liquidity": {"volume_threshold": 500000, "spread_threshold": 0.002},
            "low_liquidity": {"volume_threshold": 10000, "spread_threshold": 0.01},
        },
        "custom_curves": {
            "BTC/USDT": {"model": "square_root", "base_slippage": 0.0001},
            "ETH/USDT": {"model": "linear", "base_slippage": 0.0003},
        },
    }


@pytest.fixture
def risk_manager_with_slippage(slippage_config):
    """Fixture providing risk manager with slippage configuration."""
    config = {
        "max_position_size": Decimal("0.3"),
        "position_sizing_method": "adaptive_atr",
        "risk_per_trade": Decimal("0.02"),
        "atr_k_factor": Decimal("2.0"),
        **slippage_config,
    }
    return RiskManager(config)


@pytest.fixture
def sample_market_data_high_liquidity():
    """Fixture providing high liquidity market data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    np.random.seed(42)

    base_price = 50000.0
    returns = np.random.normal(0.0001, 0.005, 100)
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "open": np.roll(prices, 1),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
            "close": prices,
            "volume": np.random.lognormal(15, 1, 100),  # High volume
            "spread": np.random.normal(0.0005, 0.0001, 100),  # Tight spread
        },
        index=dates,
    )


@pytest.fixture
def sample_market_data_low_liquidity():
    """Fixture providing low liquidity market data."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    np.random.seed(123)

    base_price = 1000.0
    returns = np.random.normal(0.0001, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "open": np.roll(prices, 1),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            "close": prices,
            "volume": np.random.lognormal(8, 2, 100),  # Low volume
            "spread": np.random.normal(0.008, 0.003, 100),  # Wide spread
        },
        index=dates,
    )


class TestSlippageModelValidation:
    """Unit tests for slippage model validation."""

    def test_constant_slippage_calculation(self, risk_manager_with_slippage):
        """Test constant slippage model calculation."""
        manager = risk_manager_with_slippage

        # Test basic constant slippage
        slippage = manager.calculate_slippage_constant(
            order_size=Decimal("1000"), base_slippage=Decimal("0.001")
        )

        assert slippage == Decimal("0.001")
        assert isinstance(slippage, Decimal)

    def test_linear_slippage_calculation(self, risk_manager_with_slippage):
        """Test linear slippage model calculation."""
        manager = risk_manager_with_slippage

        # Test linear slippage with order size factor
        slippage = manager.calculate_slippage_linear(
            order_size=Decimal("10000"),
            base_slippage=Decimal("0.0005"),
            order_size_factor=Decimal("0.0001"),
        )

        expected = Decimal("0.0005") + (Decimal("10000") * Decimal("0.0001"))
        assert slippage == expected

    def test_square_root_slippage_calculation(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test square root slippage model calculation."""
        manager = risk_manager_with_slippage

        # Test square root slippage with volatility
        slippage = manager.calculate_slippage_square_root(
            order_size=Decimal("5000"),
            base_slippage=Decimal("0.0002"),
            volatility_factor=Decimal("0.5"),
            market_data=sample_market_data_high_liquidity,
        )

        # Should be base_slippage + sqrt(order_size) * volatility_factor * some volatility measure
        assert slippage > Decimal("0.0002")  # Should be increased by volatility
        assert isinstance(slippage, Decimal)

    def test_slippage_models_handle_zero_order_size(self, risk_manager_with_slippage):
        """Test slippage models handle zero order size gracefully."""
        manager = risk_manager_with_slippage

        # All models should handle zero order size
        constant_slippage = manager.calculate_slippage_constant(
            Decimal("0"), Decimal("0.001")
        )
        linear_slippage = manager.calculate_slippage_linear(
            Decimal("0"), Decimal("0.0005"), Decimal("0.0001")
        )
        square_root_slippage = manager.calculate_slippage_square_root(
            Decimal("0"), Decimal("0.0002"), Decimal("0.5")
        )

        assert constant_slippage == Decimal("0.001")
        assert linear_slippage == Decimal("0.0005")
        assert square_root_slippage == Decimal("0.0002")

    def test_slippage_models_handle_negative_values(self, risk_manager_with_slippage):
        """Test slippage models handle negative values gracefully."""
        manager = risk_manager_with_slippage

        # Should clamp negative results to minimum values
        linear_slippage = manager.calculate_slippage_linear(
            Decimal("-1000"), Decimal("0.0005"), Decimal("0.0001")
        )

        assert linear_slippage >= Decimal("0")


class TestLiquidityAssessment:
    """Tests for liquidity assessment functionality."""

    def test_high_liquidity_detection(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test detection of high liquidity conditions."""
        manager = risk_manager_with_slippage

        liquidity_level = manager.assess_market_liquidity(
            sample_market_data_high_liquidity
        )

        assert liquidity_level == "high_liquidity"

    def test_low_liquidity_detection(
        self, risk_manager_with_slippage, sample_market_data_low_liquidity
    ):
        """Test detection of low liquidity conditions."""
        manager = risk_manager_with_slippage

        liquidity_level = manager.assess_market_liquidity(
            sample_market_data_low_liquidity
        )

        assert liquidity_level == "low_liquidity"

    def test_liquidity_impact_on_slippage(
        self,
        risk_manager_with_slippage,
        sample_market_data_high_liquidity,
        sample_market_data_low_liquidity,
    ):
        """Test how liquidity affects slippage calculations."""
        manager = risk_manager_with_slippage

        # Calculate slippage for same order size in different liquidity conditions
        high_liq_slippage = manager.calculate_dynamic_slippage(
            order_size=Decimal("1000"),
            market_data=sample_market_data_high_liquidity,
            symbol="BTC/USDT",
        )

        low_liq_slippage = manager.calculate_dynamic_slippage(
            order_size=Decimal("1000"),
            market_data=sample_market_data_low_liquidity,
            symbol="BTC/USDT",
        )

        # Low liquidity should result in higher slippage
        assert low_liq_slippage > high_liq_slippage

    def test_custom_slippage_curves(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test custom slippage curves per trading pair."""
        manager = risk_manager_with_slippage

        # BTC/USDT should use square root model with custom base slippage
        btc_slippage = manager.calculate_dynamic_slippage(
            order_size=Decimal("1000"),
            market_data=sample_market_data_high_liquidity,
            symbol="BTC/USDT",
        )

        # ETH/USDT should use linear model with different base slippage
        eth_slippage = manager.calculate_dynamic_slippage(
            order_size=Decimal("1000"),
            market_data=sample_market_data_high_liquidity,
            symbol="ETH/USDT",
        )

        # Different symbols should have different slippage calculations
        assert btc_slippage != eth_slippage


class TestSlippageIntegration:
    """Tests for slippage integration with position sizing."""

    @patch("risk.risk_manager.RiskManager._get_current_balance")
    async def test_position_size_with_slippage(
        self,
        mock_balance,
        risk_manager_with_slippage,
        sample_market_data_high_liquidity,
    ):
        """Test position sizing that accounts for slippage."""
        mock_balance.return_value = Decimal("10000")
        manager = risk_manager_with_slippage

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="MARKET",
            amount=0,  # To be calculated
            current_price=Decimal("50000"),
            stop_loss=Decimal("49000"),
        )

        # Calculate position size with slippage consideration
        position_size = await manager.calculate_position_size_with_slippage(
            signal, sample_market_data_high_liquidity
        )

        assert position_size > 0
        assert isinstance(position_size, Decimal)

    @patch("risk.risk_manager.RiskManager._get_current_balance")
    async def test_slippage_impact_analysis(
        self,
        mock_balance,
        risk_manager_with_slippage,
        sample_market_data_high_liquidity,
    ):
        """Test slippage impact analysis for large orders."""
        mock_balance.return_value = Decimal("100000")
        manager = risk_manager_with_slippage

        # Test different order sizes
        small_order = Decimal("1000")
        large_order = Decimal("50000")

        small_slippage = manager.calculate_dynamic_slippage(
            small_order, sample_market_data_high_liquidity, "BTC/USDT"
        )

        large_slippage = manager.calculate_dynamic_slippage(
            large_order, sample_market_data_high_liquidity, "BTC/USDT"
        )

        # Large orders should have higher slippage
        assert large_slippage > small_slippage

        # Calculate impact analysis
        impact_analysis = manager.analyze_slippage_impact(
            order_size=large_order,
            market_data=sample_market_data_high_liquidity,
            symbol="BTC/USDT",
        )

        assert "estimated_slippage" in impact_analysis
        assert "liquidity_level" in impact_analysis
        assert "recommended_max_order" in impact_analysis


class TestSlippageAccuracyBenchmarks:
    """Tests for slippage calculation accuracy and benchmarks."""

    def test_slippage_calculation_precision(self, risk_manager_with_slippage):
        """Test that slippage calculations maintain precision."""
        manager = risk_manager_with_slippage

        # Test with very small values
        tiny_order = Decimal("0.0000001")
        slippage = manager.calculate_slippage_linear(
            tiny_order, Decimal("0.0005"), Decimal("0.0001")
        )

        # Should not lose precision
        assert slippage == Decimal("0.0005")  # Should be exactly base_slippage

    def test_slippage_consistency(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test that slippage calculations are consistent."""
        manager = risk_manager_with_slippage

        # Multiple calls with same parameters should give same result
        results = []
        for _ in range(10):
            slippage = manager.calculate_dynamic_slippage(
                Decimal("1000"), sample_market_data_high_liquidity, "BTC/USDT"
            )
            results.append(slippage)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_slippage_realistic_ranges(
        self,
        risk_manager_with_slippage,
        sample_market_data_high_liquidity,
        sample_market_data_low_liquidity,
    ):
        """Test that slippage values are in realistic ranges."""
        manager = risk_manager_with_slippage

        # High liquidity should have low slippage
        high_liq_slippage = manager.calculate_dynamic_slippage(
            Decimal("10000"), sample_market_data_high_liquidity, "BTC/USDT"
        )

        # Low liquidity should have higher but still reasonable slippage
        low_liq_slippage = manager.calculate_dynamic_slippage(
            Decimal("10000"), sample_market_data_low_liquidity, "BTC/USDT"
        )

        # Slippage should be between 0.01% and 5%
        assert Decimal("0.0001") <= high_liq_slippage <= Decimal("0.05")
        assert Decimal("0.0001") <= low_liq_slippage <= Decimal("0.05")
        assert low_liq_slippage > high_liq_slippage


@pytest.mark.timeout(15)
def test_slippage_calculation_performance(
    risk_manager_with_slippage, sample_market_data_high_liquidity
):
    """Test slippage calculation performance with timeout."""
    manager = risk_manager_with_slippage

    start_time = time.time()

    # Test with large number of calculations
    for i in range(10000):
        slippage = manager.calculate_dynamic_slippage(
            Decimal("1000"), sample_market_data_high_liquidity, "BTC/USDT"
        )
        assert slippage > 0

    # Should complete within timeout
    elapsed = time.time() - start_time
    assert elapsed < 10.0, f"Performance test took {elapsed:.2f}s, exceeded 10s limit"


class TestSlippageEdgeCases:
    """Tests for slippage model edge cases."""

    def test_illiquid_market_handling(self, risk_manager_with_slippage):
        """Test handling of illiquid market conditions."""
        manager = risk_manager_with_slippage

        # Create extremely illiquid market data
        dates = pd.date_range("2023-01-01", periods=10, freq="1h")
        illiquid_data = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [105.0] * 10,
                "low": [95.0] * 10,
                "close": [100.0] * 10,
                "volume": [10.0] * 10,  # Very low volume
                "spread": [0.05] * 10,  # Very wide spread (5%)
            },
            index=dates,
        )

        slippage = manager.calculate_dynamic_slippage(
            Decimal("1000"), illiquid_data, "SMALL/USDT"
        )

        # Should return high slippage for illiquid conditions
        assert slippage > Decimal("0.01")  # At least 1%

    def test_missing_market_data_handling(self, risk_manager_with_slippage):
        """Test handling when market data is missing."""
        manager = risk_manager_with_slippage

        # Should fall back to conservative slippage when no market data
        slippage = manager.calculate_dynamic_slippage(Decimal("1000"), None, "BTC/USDT")

        assert slippage > 0
        assert isinstance(slippage, Decimal)

    def test_extreme_order_sizes(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test slippage with extreme order sizes."""
        manager = risk_manager_with_slippage

        # Test with very large order size
        huge_order = Decimal("1000000")
        slippage = manager.calculate_dynamic_slippage(
            huge_order, sample_market_data_high_liquidity, "BTC/USDT"
        )

        # Should handle large orders without crashing
        assert slippage > 0
        assert slippage < Decimal("1.0")  # Should not be ridiculous

    def test_invalid_symbol_handling(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test handling of invalid or unknown symbols."""
        manager = risk_manager_with_slippage

        # Should use default slippage model for unknown symbols
        slippage = manager.calculate_dynamic_slippage(
            Decimal("1000"), sample_market_data_high_liquidity, "UNKNOWN/PAIR"
        )

        assert slippage > 0

    def test_empty_market_data_handling(self, risk_manager_with_slippage):
        """Test handling of empty market data."""
        manager = risk_manager_with_slippage

        empty_data = pd.DataFrame()

        slippage = manager.calculate_dynamic_slippage(
            Decimal("1000"), empty_data, "BTC/USDT"
        )

        # Should fall back gracefully
        assert slippage > 0


class TestSlippageMetrics:
    """Tests for slippage estimation accuracy metrics."""

    def test_slippage_accuracy_metrics_calculation(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test calculation of slippage accuracy metrics."""
        manager = risk_manager_with_slippage

        metrics = manager.calculate_slippage_accuracy_metrics(
            sample_market_data_high_liquidity, "BTC/USDT"
        )

        assert "mean_slippage" in metrics
        assert "max_slippage" in metrics
        assert "volatility_adjustment" in metrics
        assert "liquidity_score" in metrics

        # All metrics should be reasonable values
        assert all(isinstance(v, (int, float, Decimal)) for v in metrics.values())

    def test_slippage_model_comparison(
        self, risk_manager_with_slippage, sample_market_data_high_liquidity
    ):
        """Test comparison of different slippage models."""
        manager = risk_manager_with_slippage

        order_size = Decimal("5000")

        constant_slippage = manager.calculate_slippage_constant(
            order_size, Decimal("0.001")
        )
        linear_slippage = manager.calculate_slippage_linear(
            order_size, Decimal("0.0005"), Decimal("0.0001")
        )
        square_root_slippage = manager.calculate_slippage_square_root(
            order_size,
            Decimal("0.0002"),
            Decimal("0.5"),
            sample_market_data_high_liquidity,
        )

        # Different models should give different results
        slippages = [constant_slippage, linear_slippage, square_root_slippage]
        assert len(set(slippages)) > 1  # Not all the same

        # All should be positive
        assert all(s > 0 for s in slippages)


if __name__ == "__main__":
    pytest.main([__file__])
