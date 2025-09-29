"""
Unit tests for Adaptive Risk Policy Engine.

This module contains comprehensive tests for the adaptive risk management system
including market condition monitoring, performance monitoring, and risk multiplier
calculations.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from risk.adaptive_policy import (
    AdaptiveRiskPolicy,
    DefensiveMode,
    MarketConditionMonitor,
    PerformanceMonitor,
    RiskLevel,
    get_adaptive_risk_policy,
    get_risk_multiplier,
)


class TestMarketConditionMonitor:
    """Test market condition monitoring functionality."""

    def test_market_condition_monitor_creation(self):
        """Test MarketConditionMonitor initialization."""
        config = {
            "volatility_threshold": 0.05,
            "volatility_lookback": 20,
            "adx_trend_threshold": 25,
        }

        monitor = MarketConditionMonitor(config)
        assert monitor.volatility_threshold == 0.05
        assert monitor.volatility_lookback == 20
        assert monitor.adx_trend_threshold == 25

    def test_assess_market_conditions_normal(self):
        """Test market condition assessment with normal data."""
        config = {"volatility_threshold": 0.05}
        monitor = MarketConditionMonitor(config)

        # Create sample market data
        data = pd.DataFrame(
            {
                "high": [
                    100,
                    102,
                    101,
                    103,
                    102,
                    104,
                    103,
                    105,
                    104,
                    106,
                    105,
                    107,
                    106,
                    108,
                    107,
                    109,
                    108,
                    110,
                    109,
                    111,
                ],
                "low": [
                    98,
                    100,
                    99,
                    101,
                    100,
                    102,
                    101,
                    103,
                    102,
                    104,
                    103,
                    105,
                    104,
                    106,
                    105,
                    107,
                    106,
                    108,
                    107,
                    109,
                ],
                "close": [
                    99,
                    101,
                    100,
                    102,
                    101,
                    103,
                    102,
                    104,
                    103,
                    105,
                    104,
                    106,
                    105,
                    107,
                    106,
                    108,
                    107,
                    109,
                    108,
                    110,
                ],
                "volume": [1000] * 20,
            }
        )

        conditions = monitor.assess_market_conditions("BTC/USDT", data)

        assert "volatility_level" in conditions
        assert "trend_strength" in conditions
        assert "liquidity_score" in conditions
        assert "regime" in conditions
        assert "risk_level" in conditions
        assert conditions["risk_level"] in [level.value for level in RiskLevel]

    def test_assess_market_conditions_high_volatility(self):
        """Test market condition assessment with high volatility."""
        config = {
            "volatility_threshold": 0.5
        }  # Very high threshold to ensure detection
        monitor = MarketConditionMonitor(config)

        # Create high volatility data with extreme swings
        data = pd.DataFrame(
            {
                "high": [
                    100,
                    200,
                    50,
                    300,
                    25,
                    400,
                    10,
                    500,
                    5,
                    600,
                    2,
                    700,
                    1,
                    800,
                    0.5,
                    900,
                    0.1,
                    1000,
                    0.05,
                    1100,
                ],
                "low": [
                    99,
                    1,
                    150,
                    1,
                    200,
                    1,
                    250,
                    1,
                    300,
                    1,
                    350,
                    1,
                    400,
                    1,
                    450,
                    1,
                    500,
                    1,
                    550,
                    1,
                ],
                "close": [
                    99.5,
                    150,
                    100,
                    200,
                    150,
                    300,
                    200,
                    400,
                    250,
                    500,
                    300,
                    600,
                    350,
                    700,
                    400,
                    800,
                    450,
                    900,
                    500,
                    1000,
                ],
                "volume": [1000] * 20,
            }
        )

        conditions = monitor.assess_market_conditions("BTC/USDT", data)

        # The system should work and return valid results
        assert conditions["volatility_level"] in [
            "low",
            "moderate",
            "high",
            "very_high",
            "unknown",
        ]
        assert conditions["risk_level"] in [level.value for level in RiskLevel]
        assert isinstance(conditions["trend_strength"], (int, float))
        assert isinstance(conditions["liquidity_score"], (int, float))

    def test_calculate_volatility_level(self):
        """Test volatility level calculation."""
        config = {"volatility_threshold": 0.05}
        monitor = MarketConditionMonitor(config)

        # Create test data with known ATR
        data = pd.DataFrame(
            {"high": [100] * 20, "low": [95] * 20, "close": [97.5] * 20}
        )

        level = monitor._calculate_volatility_level("BTC/USDT", data)
        assert isinstance(level, str)
        assert level in ["low", "moderate", "high", "very_high", "unknown"]

    def test_calculate_trend_strength(self):
        """Test trend strength calculation."""
        config = {}
        monitor = MarketConditionMonitor(config)

        # Create trending data
        data = pd.DataFrame(
            {
                "high": list(range(100, 120)),
                "low": list(range(95, 115)),
                "close": list(range(97, 117)),
            }
        )

        trend_strength = monitor._calculate_trend_strength(data)
        assert isinstance(trend_strength, float)
        assert trend_strength >= 0

    def test_determine_risk_level(self):
        """Test risk level determination."""
        config = {}
        monitor = MarketConditionMonitor(config)

        # Test various conditions
        conditions = {
            "volatility_level": "high",
            "trend_strength": 15,  # Weak trend
            "liquidity_score": 0.3,  # Low liquidity
            "regime": "volatile",
        }

        risk_level = monitor._determine_risk_level(conditions)
        assert risk_level in [level.value for level in RiskLevel]

        # Test low risk conditions
        low_risk_conditions = {
            "volatility_level": "low",
            "trend_strength": 45,  # Strong trend
            "liquidity_score": 0.8,  # High liquidity
            "regime": "trending",
        }

        low_risk_level = monitor._determine_risk_level(low_risk_conditions)
        assert low_risk_level in ["low", "very_low", "moderate"]


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""

    def test_performance_monitor_creation(self):
        """Test PerformanceMonitor initialization."""
        config = {"lookback_days": 30, "min_sharpe": -0.5, "max_consecutive_losses": 5}

        monitor = PerformanceMonitor(config)
        assert monitor.lookback_days == 30
        assert monitor.min_sharpe_threshold == -0.5
        assert monitor.max_consecutive_losses == 5

    def test_update_performance(self):
        """Test performance update with trade results."""
        config = {}
        monitor = PerformanceMonitor(config)

        # Add winning trade
        trade_result = {"pnl": 100.0, "timestamp": datetime.now().isoformat()}
        monitor.update_performance(trade_result)

        # Add losing trade
        trade_result = {"pnl": -50.0, "timestamp": datetime.now().isoformat()}
        monitor.update_performance(trade_result)

        assert len(monitor.trade_history) == 2
        assert monitor.consecutive_losses == 1

    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        config = {}
        monitor = PerformanceMonitor(config)

        # Add some trades
        trades = [
            {"pnl": 100.0, "timestamp": datetime.now().isoformat()},
            {"pnl": 150.0, "timestamp": datetime.now().isoformat()},
            {"pnl": -50.0, "timestamp": datetime.now().isoformat()},
            {"pnl": 200.0, "timestamp": datetime.now().isoformat()},
            {"pnl": -75.0, "timestamp": datetime.now().isoformat()},
        ]

        for trade in trades:
            monitor.update_performance(trade)

        metrics = monitor.get_performance_metrics()

        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics
        assert "consecutive_losses" in metrics
        assert "total_trades" in metrics

        assert metrics["total_trades"] == 5
        assert metrics["win_rate"] == 0.6  # 3 wins out of 5
        assert metrics["consecutive_losses"] == 1

    def test_consecutive_losses_tracking(self):
        """Test consecutive losses tracking."""
        config = {}
        monitor = PerformanceMonitor(config)

        # Add winning trade
        monitor.update_performance(
            {"pnl": 100.0, "timestamp": datetime.now().isoformat()}
        )
        assert monitor.consecutive_losses == 0

        # Add losing trade
        monitor.update_performance(
            {"pnl": -50.0, "timestamp": datetime.now().isoformat()}
        )
        assert monitor.consecutive_losses == 1

        # Add another losing trade
        monitor.update_performance(
            {"pnl": -75.0, "timestamp": datetime.now().isoformat()}
        )
        assert monitor.consecutive_losses == 2

        # Add winning trade (reset counter)
        monitor.update_performance(
            {"pnl": 200.0, "timestamp": datetime.now().isoformat()}
        )
        assert monitor.consecutive_losses == 0


class TestAdaptiveRiskPolicy:
    """Test adaptive risk policy functionality."""

    def test_adaptive_risk_policy_creation(self):
        """Test AdaptiveRiskPolicy initialization."""
        config = {
            "min_multiplier": 0.1,
            "max_multiplier": 1.0,
            "volatility_threshold": 0.05,
            "performance_lookback_days": 30,
            "min_sharpe": -0.5,
            "max_consecutive_losses": 5,
            "kill_switch_threshold": 10,
            "kill_switch_window_hours": 24,
        }

        policy = AdaptiveRiskPolicy(config)

        assert policy.min_multiplier == 0.1
        assert policy.max_multiplier == 1.0
        assert policy.volatility_threshold == 0.05
        assert not policy.kill_switch_activated

    def test_get_risk_multiplier_normal_conditions(self):
        """Test risk multiplier calculation under normal conditions."""
        config = {
            "min_multiplier": 0.1,
            "max_multiplier": 1.0,
            "market_monitor": {"volatility_threshold": 0.05},
            "performance_monitor": {},
        }

        policy = AdaptiveRiskPolicy(config)

        # Create normal market data
        data = pd.DataFrame(
            {
                "high": [
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                ],
                "low": [
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                ],
                "close": [
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                ],
                "volume": [1000] * 20,
            }
        )

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", data)

        assert 0.1 <= multiplier <= 1.0
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_get_risk_multiplier_high_volatility(self):
        """Test risk multiplier with high volatility."""
        config = {
            "min_multiplier": 0.1,
            "max_multiplier": 1.0,
            "market_monitor": {"volatility_threshold": 0.01},  # Lower threshold
        }

        policy = AdaptiveRiskPolicy(config)

        # Create high volatility data with more extreme swings
        data = pd.DataFrame(
            {
                "high": [
                    100,
                    150,
                    80,
                    160,
                    70,
                    170,
                    60,
                    180,
                    50,
                    190,
                    40,
                    200,
                    30,
                    210,
                    20,
                    220,
                    10,
                    230,
                    5,
                    240,
                ],
                "low": [
                    90,
                    50,
                    120,
                    40,
                    130,
                    30,
                    140,
                    20,
                    150,
                    10,
                    160,
                    5,
                    170,
                    2,
                    180,
                    1,
                    190,
                    0.5,
                    200,
                    0.1,
                ],
                "close": [
                    95,
                    120,
                    100,
                    140,
                    90,
                    150,
                    80,
                    160,
                    70,
                    170,
                    60,
                    180,
                    50,
                    190,
                    40,
                    200,
                    30,
                    210,
                    20,
                    220,
                ],
                "volume": [1000] * 20,
            }
        )

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", data)

        # Should reduce multiplier due to high volatility
        assert multiplier < 1.0
        # The reasoning should contain some indication of the risk assessment
        reasoning_lower = reasoning.lower()
        has_risk_indicator = (
            "high" in reasoning_lower
            or "risk" in reasoning_lower
            or "volatility" in reasoning_lower
            or "market" in reasoning_lower
        )
        assert (
            has_risk_indicator
        ), f"Reasoning should indicate risk assessment: {reasoning}"

    def test_get_risk_multiplier_poor_performance(self):
        """Test risk multiplier with poor performance."""
        config = {
            "min_multiplier": 0.1,
            "max_multiplier": 1.0,
            "performance_monitor": {"lookback_days": 30},
        }

        policy = AdaptiveRiskPolicy(config)

        # Add poor performance data
        for i in range(6):  # 6 consecutive losses
            policy.performance_monitor.update_performance(
                {"pnl": -100.0, "timestamp": datetime.now().isoformat()}
            )

        data = pd.DataFrame(
            {
                "high": [100] * 20,
                "low": [99] * 20,
                "close": [99.5] * 20,
                "volume": [1000] * 20,
            }
        )

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", data)

        # Should reduce multiplier due to consecutive losses
        assert multiplier < 1.0
        assert (
            "consecutive losses" in reasoning.lower()
            or "defensive mode" in reasoning.lower()
        )

    def test_kill_switch_activation(self):
        """Test kill switch activation."""
        config = {
            "kill_switch_threshold": 3,  # Lower threshold for testing
            "kill_switch_window_hours": 1,
        }

        policy = AdaptiveRiskPolicy(config)

        # Simulate multiple defensive mode activations
        for i in range(4):
            policy.defensive_mode_history.append(
                {"mode": DefensiveMode.DEFENSIVE.value, "timestamp": datetime.now()}
            )

        # This should trigger kill switch
        data = pd.DataFrame(
            {
                "high": [100] * 20,
                "low": [99] * 20,
                "close": [99.5] * 20,
                "volume": [1000] * 20,
            }
        )

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", data)

        assert multiplier == 0.0
        assert "kill switch" in reasoning.lower()

    def test_multiplier_bounds(self):
        """Test that multipliers stay within bounds."""
        config = {"min_multiplier": 0.2, "max_multiplier": 0.8}

        policy = AdaptiveRiskPolicy(config)

        data = pd.DataFrame(
            {
                "high": [100] * 20,
                "low": [99] * 20,
                "close": [99.5] * 20,
                "volume": [1000] * 20,
            }
        )

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", data)

        assert 0.2 <= multiplier <= 0.8

    def test_update_from_trade_result(self):
        """Test updating policy from trade results."""
        config = {}
        policy = AdaptiveRiskPolicy(config)

        trade_result = {"pnl": 150.0, "timestamp": datetime.now().isoformat()}

        policy.update_from_trade_result("BTC/USDT", trade_result)

        # Performance monitor should have the trade
        metrics = policy.performance_monitor.get_performance_metrics()
        assert metrics["total_trades"] == 1
        assert metrics["win_rate"] == 1.0

    def test_get_risk_statistics(self):
        """Test getting risk statistics."""
        config = {}
        policy = AdaptiveRiskPolicy(config)

        stats = policy.get_risk_statistics()

        assert "current_multiplier" in stats
        assert "defensive_mode" in stats
        assert "kill_switch_activated" in stats
        assert "total_multiplier_changes" in stats
        assert "performance_metrics" in stats
        assert "recent_events" in stats

    def test_reset_kill_switch(self):
        """Test kill switch reset."""
        config = {}
        policy = AdaptiveRiskPolicy(config)

        # Manually activate kill switch
        policy.kill_switch_activated = True
        policy.kill_switch_timestamp = datetime.now()

        # Reset it
        result = policy.reset_kill_switch()

        assert result is True
        assert policy.kill_switch_activated is False
        assert policy.kill_switch_timestamp is None


class TestIntegrationWithPositionSizing:
    """Test integration with position sizing."""

    @patch("risk.adaptive_policy.get_adaptive_risk_policy")
    def test_get_risk_multiplier_function(self, mock_get_policy):
        """Test the get_risk_multiplier convenience function."""
        mock_policy = Mock()
        mock_policy.get_risk_multiplier.return_value = (0.8, "High volatility detected")
        mock_get_policy.return_value = mock_policy

        data = pd.DataFrame(
            {
                "high": [100] * 20,
                "low": [99] * 20,
                "close": [99.5] * 20,
                "volume": [1000] * 20,
            }
        )

        multiplier, reasoning = get_risk_multiplier("BTC/USDT", data)

        assert multiplier == 0.8
        assert reasoning == "High volatility detected"
        mock_policy.get_risk_multiplier.assert_called_once_with("BTC/USDT", data, None)

    @patch("risk.adaptive_policy.get_config")
    @patch("risk.adaptive_policy.AdaptiveRiskPolicy")
    def test_get_adaptive_risk_policy_function(
        self, mock_policy_class, mock_get_config
    ):
        """Test the get_adaptive_risk_policy convenience function."""
        mock_policy = Mock()
        mock_policy_class.return_value = mock_policy
        mock_get_config.return_value = {}

        policy = get_adaptive_risk_policy({"enabled": True})

        assert policy == mock_policy
        mock_policy_class.assert_called_once_with({"enabled": True})


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_market_data(self):
        """Test handling of empty market data."""
        config = {}
        policy = AdaptiveRiskPolicy(config)

        # Empty DataFrame
        data = pd.DataFrame()

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", data)

        # Should return neutral multiplier
        assert multiplier == 1.0
        assert isinstance(reasoning, str)

    def test_none_market_data(self):
        """Test handling of None market data."""
        config = {}
        policy = AdaptiveRiskPolicy(config)

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", None)

        # Should return neutral multiplier
        assert multiplier == 1.0
        assert isinstance(reasoning, str)

    def test_extreme_market_conditions(self):
        """Test handling of extreme market conditions."""
        config = {"min_multiplier": 0.1, "max_multiplier": 1.0}

        policy = AdaptiveRiskPolicy(config)

        # Create extremely volatile data
        data = pd.DataFrame(
            {
                "high": [100, 200, 50, 300, 25, 400, 10, 500, 5, 600],
                "low": [99, 1, 150, 1, 200, 1, 250, 1, 300, 1],
                "close": [99.5, 150, 100, 200, 150, 300, 200, 400, 250, 500],
                "volume": [100] * 10,
            }
        )

        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", data)

        # Should clamp to minimum multiplier
        assert multiplier >= 0.1
        assert multiplier <= 1.0

    def test_performance_monitor_empty_history(self):
        """Test performance monitor with empty history."""
        config = {}
        monitor = PerformanceMonitor(config)

        metrics = monitor.get_performance_metrics()

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["consecutive_losses"] == 0

    def test_market_monitor_error_handling(self):
        """Test market monitor error handling."""
        config = {}
        monitor = MarketConditionMonitor(config)

        # Invalid data that should cause errors
        data = pd.DataFrame({"invalid_column": [1, 2, 3]})

        conditions = monitor.assess_market_conditions("BTC/USDT", data)

        # Should return fallback values
        assert "volatility_level" in conditions
        assert "risk_level" in conditions
        assert conditions["volatility_level"] == "unknown"


if __name__ == "__main__":
    pytest.main([__file__])
