"""
Unit tests for strategy selector module.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from strategies.regime.strategy_selector import (
    MarketRegime,
    MarketStateAnalyzer,
    MLBasedSelector,
    RuleBasedSelector,
    StrategyPerformance,
    StrategySelector,
    select_strategy,
    update_strategy_performance,
)


class TestMarketStateAnalyzer:
    """Test MarketStateAnalyzer."""

    def test_analyze_market_state_trending_up(self):
        """Test analysis of up-trending market."""
        # Create trending up data
        data = pd.DataFrame(
            {
                "open": list(range(10, 50)),
                "high": [i + 3 for i in range(10, 50)],  # Higher highs
                "low": [i - 1 for i in range(10, 50)],  # Higher lows
                "close": [i + 2 for i in range(10, 50)],  # Consistently higher
                "volume": [1000] * 40,
            }
        )

        analyzer = MarketStateAnalyzer()
        regime = analyzer.analyze_market_state(data)
        assert regime in [MarketRegime.TRENDING, MarketRegime.SIDEWAYS]

    def test_analyze_market_state_sideways(self):
        """Test analysis of sideways market."""
        # Create sideways data
        data = pd.DataFrame(
            {
                "open": [10] * 40,
                "high": [11] * 40,
                "low": [9] * 40,
                "close": [10] * 40,
                "volume": [1000] * 40,
            }
        )

        analyzer = MarketStateAnalyzer()
        regime = analyzer.analyze_market_state(data)
        assert regime == MarketRegime.SIDEWAYS

    def test_analyze_market_state_insufficient_data(self):
        """Test analysis with insufficient data."""
        data = pd.DataFrame(
            {
                "open": [10, 11],
                "high": [12, 13],
                "low": [8, 9],
                "close": [11, 12],
                "volume": [1000, 1100],
            }
        )

        analyzer = MarketStateAnalyzer()
        regime = analyzer.analyze_market_state(data)
        assert regime == MarketRegime.SIDEWAYS


class TestStrategyPerformance:
    """Test StrategyPerformance."""

    def test_init(self):
        """Test initialization."""
        perf = StrategyPerformance("TestStrategy")
        assert perf.strategy_name == "TestStrategy"
        assert perf.total_trades == 0
        assert perf.win_rate == 0.0

    def test_update_trade_win(self):
        """Test updating with winning trade."""
        perf = StrategyPerformance("TestStrategy")
        perf.update_trade(100.0, 0.05, True)

        assert perf.total_trades == 1
        assert perf.winning_trades == 1
        assert perf.total_pnl == 100.0
        assert perf.win_rate == 1.0
        assert perf.avg_win == 100.0

    def test_update_trade_loss(self):
        """Test updating with losing trade."""
        perf = StrategyPerformance("TestStrategy")
        perf.update_trade(-50.0, -0.03, False)

        assert perf.total_trades == 1
        assert perf.losing_trades == 1
        assert perf.total_pnl == -50.0
        assert perf.win_rate == 0.0
        assert perf.avg_loss == 50.0

    def test_get_metrics(self):
        """Test getting performance metrics."""
        perf = StrategyPerformance("TestStrategy")
        perf.update_trade(100.0, 0.05, True)
        perf.update_trade(-50.0, -0.03, False)

        metrics = perf.get_metrics()
        assert metrics["strategy_name"] == "TestStrategy"
        assert metrics["total_trades"] == 2
        assert metrics["win_rate"] == 0.5
        assert metrics["total_pnl"] == 50.0

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        perf = StrategyPerformance("TestStrategy")

        # Add some trades with returns
        for i in range(10):
            perf.update_trade(
                10.0 if i % 2 == 0 else -5.0, 0.01 if i % 2 == 0 else -0.005, i % 2 == 0
            )

        sharpe = perf.calculate_sharpe_ratio()
        assert isinstance(sharpe, float)


class TestRuleBasedSelector:
    """Test RuleBasedSelector."""

    def test_init(self):
        """Test initialization."""
        config = {"adx_trend_threshold": 30}
        selector = RuleBasedSelector(config)
        assert selector.adx_trend_threshold == 30

    @patch("strategies.regime.strategy_selector.EMACrossStrategy")
    @patch("strategies.regime.strategy_selector.RSIStrategy")
    def test_select_strategy_trending(self, mock_rsi, mock_ema):
        """Test strategy selection for trending market."""
        # Mock the strategy classes
        mock_rsi.__name__ = "RSIStrategy"
        mock_ema.__name__ = "EMACrossStrategy"

        selector = RuleBasedSelector()

        # Create trending data
        data = pd.DataFrame(
            {
                "open": list(range(10, 50)),
                "high": [i + 3 for i in range(10, 50)],
                "low": [i - 1 for i in range(10, 50)],
                "close": [i + 2 for i in range(10, 50)],
                "volume": [1000] * 40,
            }
        )

        available_strategies = [mock_ema, mock_rsi]
        selected = selector.select_strategy(data, available_strategies)

        # Should select EMA for trending markets
        assert selected == mock_ema

    @patch("strategies.regime.strategy_selector.get_market_regime_detector")
    @patch("strategies.regime.strategy_selector.EMACrossStrategy")
    @patch("strategies.regime.strategy_selector.RSIStrategy")
    def test_select_strategy_sideways(self, mock_rsi, mock_ema, mock_get_detector):
        """Test strategy selection for sideways market."""
        # Note: patch order means mock_rsi is RSIStrategy, mock_ema is EMACrossStrategy
        mock_rsi.__name__ = "RSIStrategy"
        mock_ema.__name__ = "EMACrossStrategy"

        # Mock the regime detector to return SIDEWAYS
        from strategies.regime.market_regime import MarketRegime

        mock_detector = MagicMock()
        mock_result = MagicMock()
        mock_result.regime = MarketRegime.SIDEWAYS
        mock_detector.detect_regime.return_value = mock_result
        mock_get_detector.return_value = mock_detector

        selector = RuleBasedSelector()

        # Create sideways data
        data = pd.DataFrame(
            {
                "open": [10] * 40,
                "high": [11] * 40,
                "low": [9] * 40,
                "close": [10] * 40,
                "volume": [1000] * 40,
            }
        )

        available_strategies = [mock_ema, mock_rsi]
        selected = selector.select_strategy(data, available_strategies)

        # Should select RSI for sideways markets
        assert selected == mock_rsi

    def test_select_strategy_empty_data(self):
        """Test strategy selection with empty data."""
        selector = RuleBasedSelector()
        data = pd.DataFrame()
        available_strategies = []

        selected = selector.select_strategy(data, available_strategies)
        assert selected is None


class TestMLBasedSelector:
    """Test MLBasedSelector."""

    def test_init(self):
        """Test initialization."""
        config = {"learning_rate": 0.2}
        selector = MLBasedSelector(config)
        assert selector.learning_rate == 0.2

    @patch("strategies.regime.strategy_selector.EMACrossStrategy")
    @patch("strategies.regime.strategy_selector.RSIStrategy")
    def test_select_strategy(self, mock_rsi, mock_ema):
        """Test ML-based strategy selection."""
        mock_ema.__name__ = "EMACrossStrategy"
        mock_rsi.__name__ = "RSIStrategy"

        selector = MLBasedSelector()

        # Create mock performance data with proper attributes
        from strategies.regime.strategy_selector import StrategyPerformance

        perf_ema = StrategyPerformance("EMACrossStrategy")
        perf_rsi = StrategyPerformance("RSIStrategy")

        # Add some trades to meet minimum threshold
        for _ in range(15):
            perf_ema.update_trade(10.0, 0.01, True)
            perf_rsi.update_trade(-5.0, -0.005, False)

        performances = {"EMACrossStrategy": perf_ema, "RSIStrategy": perf_rsi}

        data = pd.DataFrame({"close": [10, 11, 12]})
        available_strategies = [mock_ema, mock_rsi]

        selected = selector.select_strategy(data, available_strategies, performances)
        assert selected in available_strategies

    def test_update_weights(self):
        """Test weight updates based on performance."""
        selector = MLBasedSelector()
        selector.strategy_weights = {"StrategyA": 1.0, "StrategyB": 1.0}

        # Mock performances
        performances = {"StrategyA": MagicMock(), "StrategyB": MagicMock()}
        performances["StrategyA"].total_trades = 20
        performances["StrategyA"].win_rate = 0.7
        performances["StrategyB"].total_trades = 20
        performances["StrategyB"].win_rate = 0.3

        selector._update_weights(performances)

        # StrategyA should have higher weight
        assert (
            selector.strategy_weights["StrategyA"]
            > selector.strategy_weights["StrategyB"]
        )


class TestStrategySelector:
    """Test StrategySelector."""

    @patch("strategies.regime.strategy_selector.get_config")
    def test_init(self, mock_get_config):
        """Test initialization."""
        mock_get_config.return_value = {
            "enabled": True,
            "mode": "rule_based",
            "ensemble": False,
        }

        selector = StrategySelector()
        assert selector.enabled is True
        assert selector.mode == "rule_based"
        assert selector.ensemble is False

    @patch("strategies.regime.strategy_selector.get_config")
    @patch("strategies.regime.strategy_selector.EMACrossStrategy")
    @patch("strategies.regime.strategy_selector.RSIStrategy")
    def test_select_strategy_rule_based(self, mock_rsi, mock_ema, mock_get_config):
        """Test rule-based strategy selection."""
        mock_get_config.return_value = {
            "enabled": True,
            "mode": "rule_based",
            "ensemble": False,
        }
        mock_rsi.__name__ = "RSIStrategy"
        mock_ema.__name__ = "EMACrossStrategy"

        selector = StrategySelector()

        # Create trending data
        data = pd.DataFrame(
            {
                "open": list(range(10, 50)),
                "high": [i + 3 for i in range(10, 50)],
                "low": [i - 1 for i in range(10, 50)],
                "close": [i + 2 for i in range(10, 50)],
                "volume": [1000] * 40,
            }
        )

        selected = selector.select_strategy(data)
        assert selected in [mock_ema, mock_rsi]

    @patch("strategies.regime.strategy_selector.get_config")
    def test_select_strategy_disabled(self, mock_get_config):
        """Test strategy selection when disabled."""
        mock_get_config.return_value = {"enabled": False}

        selector = StrategySelector()
        data = pd.DataFrame({"close": [10, 11, 12]})

        selected = selector.select_strategy(data)
        # When disabled, it returns the first available strategy as fallback
        assert selected is not None

    @patch("strategies.regime.strategy_selector.get_config")
    def test_update_performance(self, mock_get_config):
        """Test performance update."""
        mock_get_config.return_value = {"enabled": True}

        selector = StrategySelector()
        selector.update_performance("TestStrategy", 100.0, 0.05, True)

        perf = selector.strategy_performances.get("TestStrategy")
        assert perf is not None
        assert perf.total_trades == 1
        assert perf.winning_trades == 1

    @patch("strategies.regime.strategy_selector.get_config")
    def test_get_strategy_performance(self, mock_get_config):
        """Test getting strategy performance."""
        mock_get_config.return_value = {"enabled": True}

        selector = StrategySelector()
        selector.update_performance("TestStrategy", 100.0, 0.05, True)

        perf = selector.get_strategy_performance("TestStrategy")
        assert perf["total_trades"] == 1
        assert perf["win_rate"] == 1.0

    @patch("strategies.regime.strategy_selector.get_config")
    def test_save_load_performance_history(self, mock_get_config):
        """Test saving and loading performance history."""
        mock_get_config.return_value = {"enabled": True}

        selector = StrategySelector()
        selector.update_performance("TestStrategy", 100.0, 0.05, True)

        # Save performance
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            selector.save_performance_history(temp_path)
            assert os.path.exists(temp_path)

            # Load performance
            selector.load_performance_history(temp_path)

            perf = selector.get_strategy_performance("TestStrategy")
            assert perf["total_trades"] == 1

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestGlobalFunctions:
    """Test global convenience functions."""

    @patch("strategies.regime.strategy_selector.get_strategy_selector")
    def test_select_strategy_global(self, mock_get_selector):
        """Test global select_strategy function."""
        mock_selector = MagicMock()
        mock_selector.select_strategy.return_value = "TestStrategy"
        mock_get_selector.return_value = mock_selector

        data = pd.DataFrame({"close": [10, 11, 12]})
        result = select_strategy(data)

        assert result == "TestStrategy"
        mock_selector.select_strategy.assert_called_once_with(data)

    @patch("strategies.regime.strategy_selector.get_strategy_selector")
    def test_update_strategy_performance_global(self, mock_get_selector):
        """Test global update_strategy_performance function."""
        mock_selector = MagicMock()
        mock_get_selector.return_value = mock_selector

        update_strategy_performance("TestStrategy", 100.0, 0.05, True)

        mock_selector.update_performance.assert_called_once_with(
            "TestStrategy", 100.0, 0.05, True
        )


class TestErrorHandling:
    """Test error handling."""

    @patch("strategies.regime.strategy_selector.get_config")
    def test_strategy_selector_init_error_handling(self, mock_get_config):
        """Test initialization error handling."""
        mock_get_config.return_value = {"enabled": True}

        # This should not raise an error even if strategies can't be loaded
        selector = StrategySelector()
        assert selector is not None

    @patch("strategies.regime.strategy_selector.get_config")
    def test_market_state_analyzer_error_handling(self, mock_get_config):
        """Test market state analyzer error handling."""
        analyzer = MarketStateAnalyzer()

        # Empty data should return SIDEWAYS
        data = pd.DataFrame()
        regime = analyzer.analyze_market_state(data)
        assert regime == MarketRegime.SIDEWAYS

    def test_strategy_performance_edge_cases(self):
        """Test strategy performance edge cases."""
        perf = StrategyPerformance("TestStrategy")

        # No trades
        sharpe = perf.calculate_sharpe_ratio()
        assert sharpe == 0.0

        # Single trade
        perf.update_trade(100.0, 0.05, True)
        sharpe = perf.calculate_sharpe_ratio()
        assert sharpe == 0.0  # Need at least 2 trades


class TestIntegration:
    """Test integration scenarios."""

    @patch("strategies.regime.strategy_selector.get_config")
    @patch("strategies.regime.strategy_selector.EMACrossStrategy")
    @patch("strategies.regime.strategy_selector.RSIStrategy")
    def test_full_selection_workflow(self, mock_rsi, mock_ema, mock_get_config):
        """Test full strategy selection workflow."""
        mock_get_config.return_value = {
            "enabled": True,
            "mode": "rule_based",
            "ensemble": False,
        }
        mock_rsi.__name__ = "RSIStrategy"
        mock_ema.__name__ = "EMACrossStrategy"

        selector = StrategySelector()

        # Create market data
        data = pd.DataFrame(
            {
                "open": list(range(10, 50)),
                "high": [i + 3 for i in range(10, 50)],
                "low": [i - 1 for i in range(10, 50)],
                "close": [i + 2 for i in range(10, 50)],
                "volume": [1000] * 40,
            }
        )

        # Select strategy
        selected = selector.select_strategy(data)
        assert selected is not None

        # Update performance
        selector.update_performance(selected.__name__, 100.0, 0.05, True)

        # Check performance
        perf = selector.get_strategy_performance(selected.__name__)
        assert perf["total_trades"] == 1
        assert perf["win_rate"] == 1.0

    @patch("strategies.regime.strategy_selector.get_config")
    def test_ensemble_mode(self, mock_get_config):
        """Test ensemble mode selection."""
        mock_get_config.return_value = {
            "enabled": True,
            "mode": "rule_based",
            "ensemble": True,
        }

        selector = StrategySelector()

        # Create market data
        data = pd.DataFrame(
            {
                "open": [10] * 40,
                "high": [11] * 40,
                "low": [9] * 40,
                "close": [10] * 40,
                "volume": [1000] * 40,
            }
        )

        # Select strategies for ensemble
        selected = selector.select_strategies_ensemble(data, max_strategies=2)
        assert isinstance(selected, list)
        assert len(selected) <= 2
