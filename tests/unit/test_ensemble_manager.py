"""
Unit tests for EnsembleManager functionality.
"""

import pytest
import pandas as pd
import numpy as np
from decimal import Decimal
from unittest.mock import MagicMock, patch
from core.ensemble_manager import (
    EnsembleManager,
    VotingMode,
    EnsembleDecision,
    StrategySignal
)
from core.contracts import TradingSignal, SignalType, SignalStrength


class MockStrategy:
    """Mock strategy for testing."""

    def __init__(self, strategy_id, signals=None):
        self.strategy_id = strategy_id
        self.signals = signals or []

    def generate_signals(self, market_data):
        """Mock signal generation."""
        return self.signals


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "ohlcv": pd.DataFrame({
            "timestamp": [1234567890],
            "open": [50000],
            "high": [51000],
            "low": [49000],
            "close": [50500],
            "volume": [100]
        })
    }


@pytest.fixture
def sample_signals():
    """Sample trading signals for testing."""
    return [
        TradingSignal(
            strategy_id="strategy1",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            current_price=Decimal("50000"),
            timestamp=1234567890,
            stop_loss=Decimal("49000"),
            take_profit=Decimal("52000"),
            metadata={"confidence": 0.8}
        ),
            TradingSignal(
                strategy_id="strategy2",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.MODERATE,
                order_type="market",
                amount=Decimal("1.0"),
                price=Decimal("50000"),
                current_price=Decimal("50000"),
                timestamp=1234567890,
                stop_loss=Decimal("49000"),
                take_profit=Decimal("52000"),
                metadata={"confidence": 0.7}
            ),
        TradingSignal(
            strategy_id="strategy3",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.WEAK,
            order_type="market",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            current_price=Decimal("50000"),
            timestamp=1234567890,
            stop_loss=Decimal("51000"),
            take_profit=Decimal("48000"),
            metadata={"confidence": 0.6}
        )
    ]


@pytest.fixture
def ensemble_manager():
    """EnsembleManager instance for testing."""
    config = {
        "enabled": True,
        "mode": "weighted_vote",
        "dynamic_weights": True,
        "strategies": [
            {"id": "strategy1", "weight": 0.4},
            {"id": "strategy2", "weight": 0.3},
            {"id": "strategy3", "weight": 0.3}
        ],
        "thresholds": {
            "confidence": 0.6,
            "vote_ratio": 0.66
        }
    }
    return EnsembleManager(config)


class TestEnsembleManager:
    """Test cases for EnsembleManager."""

    def test_initialization(self, ensemble_manager):
        """Test EnsembleManager initialization."""
        assert ensemble_manager.enabled is True
        assert ensemble_manager.voting_mode == VotingMode.WEIGHTED_VOTE
        assert ensemble_manager.dynamic_weights is True
        assert len(ensemble_manager.strategy_weights) == 3
        assert "strategy1" in ensemble_manager.strategy_weights

    def test_register_strategy(self, ensemble_manager):
        """Test strategy registration."""
        mock_strategy = MockStrategy("test_strategy")
        ensemble_manager.register_strategy("test_strategy", mock_strategy)

        assert "test_strategy" in ensemble_manager.strategies
        assert ensemble_manager.strategies["test_strategy"] is mock_strategy
        assert "test_strategy" in ensemble_manager.strategy_weights

    def test_get_ensemble_signal_disabled(self):
        """Test ensemble signal when disabled."""
        config = {"enabled": False}
        manager = EnsembleManager(config)

        result = manager.get_ensemble_signal({})
        assert result is None

    def test_get_ensemble_signal_no_strategies(self, ensemble_manager):
        """Test ensemble signal with no registered strategies."""
        # Clear strategies
        ensemble_manager.strategies.clear()

        result = ensemble_manager.get_ensemble_signal({})
        assert result is None

    def test_extract_confidence_from_metadata(self, ensemble_manager, sample_signals):
        """Test confidence extraction from signal metadata."""
        signal = sample_signals[0]  # Has confidence: 0.8

        confidence = ensemble_manager._extract_confidence(signal, {})
        assert confidence == 0.8

    def test_extract_confidence_from_market_data(self, ensemble_manager, sample_signals):
        """Test confidence extraction from market data."""
        signal = sample_signals[0]
        market_data = {"ml_prediction": {"confidence": 0.9}}

        confidence = ensemble_manager._extract_confidence(signal, market_data)
        assert confidence == 0.9

    def test_extract_confidence_default_from_strength(self, ensemble_manager):
        """Test default confidence based on signal strength."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        confidence = ensemble_manager._extract_confidence(signal, {})
        assert confidence == 0.9  # Strong = 0.9

    def test_majority_vote_buy_consensus(self, ensemble_manager, sample_signals):
        """Test majority vote with buy consensus."""
        ensemble_manager.voting_mode = VotingMode.MAJORITY_VOTE

        # Create signals: 2 buy, 1 sell
        strategy_signals = [
            StrategySignal("s1", sample_signals[0], 0.8, 1.0),  # BUY
            StrategySignal("s2", sample_signals[0], 0.7, 1.0),  # BUY
            StrategySignal("s3", sample_signals[2], 0.6, 1.0),  # SELL
        ]

        result = ensemble_manager._majority_vote(strategy_signals)

        assert result.decision == EnsembleDecision.BUY
        assert result.final_signal is not None
        assert result.final_signal.signal_type == SignalType.ENTRY_LONG
        assert result.vote_counts["buy"] == 2
        assert result.vote_counts["sell"] == 1

    def test_majority_vote_no_consensus(self, ensemble_manager):
        """Test majority vote with no consensus."""
        ensemble_manager.voting_mode = VotingMode.MAJORITY_VOTE

        # Create signals with equal votes
        buy_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )
        sell_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        strategy_signals = [
            StrategySignal("s1", buy_signal, 0.8, 1.0),  # BUY
            StrategySignal("s2", sell_signal, 0.8, 1.0),  # SELL
        ]

        result = ensemble_manager._majority_vote(strategy_signals)

        assert result.decision == EnsembleDecision.NO_CONSENSUS
        assert result.final_signal is None

    def test_weighted_vote_high_weight_dominates(self, ensemble_manager):
        """Test weighted vote where high weight strategy dominates."""
        ensemble_manager.voting_mode = VotingMode.WEIGHTED_VOTE

        buy_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )
        sell_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        # High weight on sell, low on buy
        strategy_signals = [
            StrategySignal("s1", buy_signal, 0.8, 0.2),   # Low weight BUY
            StrategySignal("s2", sell_signal, 0.6, 0.8),  # High weight SELL
        ]

        result = ensemble_manager._weighted_vote(strategy_signals)

        assert result.decision == EnsembleDecision.SELL
        assert result.final_signal is not None
        assert result.final_signal.signal_type == SignalType.ENTRY_SHORT

    def test_confidence_average_high_confidence_buy(self, ensemble_manager):
        """Test confidence average with high confidence buy signals."""
        ensemble_manager.voting_mode = VotingMode.CONFIDENCE_AVERAGE

        buy_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        strategy_signals = [
            StrategySignal("s1", buy_signal, 0.8, 1.0),  # High confidence BUY
            StrategySignal("s2", buy_signal, 0.9, 1.0),  # High confidence BUY
        ]

        result = ensemble_manager._confidence_average(strategy_signals)

        assert result.decision == EnsembleDecision.BUY
        assert result.final_signal is not None
        assert result.confidence_score > 0.8

    def test_confidence_average_low_confidence_no_trade(self, ensemble_manager):
        """Test confidence average with low confidence signals."""
        ensemble_manager.voting_mode = VotingMode.CONFIDENCE_AVERAGE
        ensemble_manager.thresholds["confidence"] = 0.8  # High threshold

        buy_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        strategy_signals = [
            StrategySignal("s1", buy_signal, 0.5, 1.0),  # Low confidence
            StrategySignal("s2", buy_signal, 0.6, 1.0),  # Low confidence
        ]

        result = ensemble_manager._confidence_average(strategy_signals)

        assert result.decision == EnsembleDecision.NO_CONSENSUS
        assert result.final_signal is None
        assert result.confidence_score < 0.8

    def test_create_ensemble_signal(self, ensemble_manager, sample_signals):
        """Test creation of ensemble signal."""
        strategy_signals = [
            StrategySignal("s1", sample_signals[0], 0.8, 0.5),
            StrategySignal("s2", sample_signals[1], 0.7, 0.5),
        ]

        result = ensemble_manager._create_ensemble_signal(
            strategy_signals, SignalType.ENTRY_LONG
        )

        assert result is not None
        assert result.strategy_id == "ensemble"
        assert result.signal_type == SignalType.ENTRY_LONG
        assert "ensemble" in result.metadata
        assert result.metadata["contributing_strategies"] == ["s1", "s2"]
        assert result.metadata["voting_mode"] == "weighted_vote"

    def test_update_weights_dynamic_enabled(self, ensemble_manager):
        """Test dynamic weight updates."""
        ensemble_manager.dynamic_weights = True

        performance_metrics = {
            "strategy1": {
                "sharpe_ratio": 2.0,
                "win_rate": 0.8,
                "profit_factor": 1.5
            },
            "strategy2": {
                "sharpe_ratio": 0.5,
                "win_rate": 0.4,
                "profit_factor": 0.8
            }
        }

        ensemble_manager.update_weights(performance_metrics)

        # Strategy1 should have higher weight than strategy2
        assert ensemble_manager.strategy_weights["strategy1"] > ensemble_manager.strategy_weights["strategy2"]

    def test_update_weights_dynamic_disabled(self, ensemble_manager):
        """Test weight updates when dynamic weights are disabled."""
        ensemble_manager.dynamic_weights = False

        original_weights = ensemble_manager.strategy_weights.copy()

        performance_metrics = {
            "strategy1": {"sharpe_ratio": 2.0, "win_rate": 0.8, "profit_factor": 1.5}
        }

        ensemble_manager.update_weights(performance_metrics)

        # Weights should remain unchanged
        assert ensemble_manager.strategy_weights == original_weights

    def test_calculate_weight_from_metrics(self, ensemble_manager):
        """Test weight calculation from performance metrics."""
        metrics = {
            "sharpe_ratio": 2.0,
            "win_rate": 0.8,
            "profit_factor": 1.5
        }

        weight = ensemble_manager._calculate_weight_from_metrics(metrics)

        # Should be a positive weight
        assert weight > 0
        assert weight <= 1.0

    def test_calculate_weight_minimum_weight(self, ensemble_manager):
        """Test minimum weight enforcement."""
        metrics = {
            "sharpe_ratio": -5.0,  # Very bad
            "win_rate": 0.1,       # Very bad
            "profit_factor": 0.1   # Very bad
        }

        weight = ensemble_manager._calculate_weight_from_metrics(metrics)

        # Should be at least minimum weight
        assert weight >= 0.1

    def test_set_voting_mode(self, ensemble_manager):
        """Test voting mode changes."""
        ensemble_manager.set_voting_mode(VotingMode.CONFIDENCE_AVERAGE)

        assert ensemble_manager.voting_mode == VotingMode.CONFIDENCE_AVERAGE

    def test_enable_ensemble(self, ensemble_manager):
        """Test ensemble enable/disable."""
        ensemble_manager.enable_ensemble(False)
        assert ensemble_manager.enabled is False

        ensemble_manager.enable_ensemble(True)
        assert ensemble_manager.enabled is True

    def test_get_strategy_performance(self, ensemble_manager):
        """Test getting strategy performance metrics."""
        performance = ensemble_manager.get_strategy_performance()

        assert isinstance(performance, dict)
        assert "strategy1" in performance

    def test_get_strategy_weights(self, ensemble_manager):
        """Test getting strategy weights."""
        weights = ensemble_manager.get_strategy_weights()

        assert isinstance(weights, dict)
        assert "strategy1" in weights
        assert weights["strategy1"] == 0.4

    @patch('core.ensemble_manager.trade_logger')
    def test_log_ensemble_decision(self, mock_logger, ensemble_manager, sample_signals):
        """Test ensemble decision logging."""
        from core.ensemble_manager import EnsembleResult

        result = EnsembleResult(
            decision=EnsembleDecision.BUY,
            final_signal=sample_signals[0],
            contributing_strategies=["s1", "s2"],
            vote_counts={"buy": 2, "sell": 0, "hold": 0},
            total_weight=2.0,
            confidence_score=0.8,
            voting_mode=VotingMode.WEIGHTED_VOTE,
            strategy_weights={"s1": 0.6, "s2": 0.4}
        )

        ensemble_manager._log_ensemble_decision(result)

        # Verify logging was called
        mock_logger.trade.assert_called_once()

    def test_make_ensemble_decision_unknown_mode(self, ensemble_manager):
        """Test ensemble decision with unknown voting mode."""
        ensemble_manager.voting_mode = "unknown_mode"

        strategy_signals = [
            StrategySignal("s1", None, 0.8, 1.0)
        ]

        result = ensemble_manager._make_ensemble_decision(strategy_signals)

        assert result.decision == EnsembleDecision.NO_CONSENSUS
        assert result.final_signal is None

    def test_majority_vote_tie_handling(self, ensemble_manager):
        """Test majority vote tie handling."""
        ensemble_manager.voting_mode = VotingMode.MAJORITY_VOTE

        buy_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )
        sell_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_SHORT,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        # Create perfect tie
        strategy_signals = [
            StrategySignal("s1", buy_signal, 0.8, 1.0),   # BUY
            StrategySignal("s2", sell_signal, 0.8, 1.0),  # SELL
        ]

        result = ensemble_manager._majority_vote(strategy_signals)

        assert result.decision == EnsembleDecision.NO_CONSENSUS
        assert result.final_signal is None

    def test_weighted_vote_with_zero_weights(self, ensemble_manager):
        """Test weighted vote with zero weights."""
        ensemble_manager.voting_mode = VotingMode.WEIGHTED_VOTE

        buy_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        strategy_signals = [
            StrategySignal("s1", buy_signal, 0.8, 0.0),   # Zero weight
            StrategySignal("s2", buy_signal, 0.6, 0.0),   # Zero weight
        ]

        result = ensemble_manager._weighted_vote(strategy_signals)

        assert result.decision == EnsembleDecision.NO_CONSENSUS
        assert result.final_signal is None

    def test_confidence_average_empty_confidences(self, ensemble_manager):
        """Test confidence average with empty confidence lists."""
        ensemble_manager.voting_mode = VotingMode.CONFIDENCE_AVERAGE

        # Create signals that will result in empty confidence lists
        # by having signals that don't match ENTRY_LONG or ENTRY_SHORT
        exit_signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.EXIT_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        strategy_signals = [
            StrategySignal("s1", exit_signal, 0.8, 1.0),   # EXIT (not entry)
        ]

        result = ensemble_manager._confidence_average(strategy_signals)

        # Since there are no entry signals, all confidence lists will be empty
        # and the method should handle this gracefully
        assert result.decision == EnsembleDecision.NO_CONSENSUS
        assert result.final_signal is None

    def test_create_ensemble_signal_no_relevant_signals(self, ensemble_manager):
        """Test ensemble signal creation with no relevant signals."""
        strategy_signals = [
            StrategySignal("s1", None, 0.8, 1.0),  # No signal
        ]

        result = ensemble_manager._create_ensemble_signal(
            strategy_signals, SignalType.ENTRY_LONG
        )

        assert result is None

    def test_extract_confidence_invalid_values(self, ensemble_manager):
        """Test confidence extraction with invalid values."""
        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0"),
            metadata={"confidence": "invalid"}  # Invalid confidence
        )

        confidence = ensemble_manager._extract_confidence(signal, {})
        assert confidence == 0.9  # Should fall back to strength-based default

    def test_register_strategy_overwrites_existing(self, ensemble_manager):
        """Test registering a strategy that already exists."""
        mock_strategy1 = MockStrategy("test_strategy")
        mock_strategy2 = MockStrategy("test_strategy")

        ensemble_manager.register_strategy("test_strategy", mock_strategy1)
        ensemble_manager.register_strategy("test_strategy", mock_strategy2)

        # Should overwrite with new strategy
        assert ensemble_manager.strategies["test_strategy"] is mock_strategy2

    def test_get_ensemble_signal_with_strategy_errors(self, ensemble_manager, sample_market_data):
        """Test ensemble signal generation when strategies raise errors."""
        # Register a strategy that will raise an error
        class FailingStrategy:
            def generate_signals(self, market_data):
                raise Exception("Strategy error")

        ensemble_manager.register_strategy("failing_strategy", FailingStrategy())

        # Should handle the error gracefully and continue
        result = ensemble_manager.get_ensemble_signal(sample_market_data)
        # Result depends on other strategies, but should not crash
        assert result is None or isinstance(result, TradingSignal)

    def test_dynamic_weights_calculation_edge_cases(self, ensemble_manager):
        """Test dynamic weight calculation with edge case metrics."""
        # Test with missing metrics
        metrics = {}
        weight = ensemble_manager._calculate_weight_from_metrics(metrics)
        assert weight >= 0.1  # Should get minimum weight

        # Test with extreme values
        metrics = {
            "sharpe_ratio": 10.0,    # Very good
            "win_rate": 1.0,         # Perfect
            "profit_factor": 5.0     # Excellent
        }
        weight = ensemble_manager._calculate_weight_from_metrics(metrics)
        assert weight > 0

        # Test with negative values
        metrics = {
            "sharpe_ratio": -2.0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }
        weight = ensemble_manager._calculate_weight_from_metrics(metrics)
        assert weight >= 0.1  # Should be clamped to minimum

    def test_voting_mode_enum_values(self):
        """Test that all voting mode enum values are handled."""
        for mode in VotingMode:
            config = {"enabled": True, "mode": mode.value}
            manager = EnsembleManager(config)
            assert manager.voting_mode == mode

    def test_ensemble_result_dataclass(self):
        """Test EnsembleResult dataclass creation and attributes."""
        from core.ensemble_manager import EnsembleResult

        result = EnsembleResult(
            decision=EnsembleDecision.BUY,
            final_signal=None,
            contributing_strategies=["s1", "s2"],
            vote_counts={"buy": 2, "sell": 0, "hold": 0},
            total_weight=2.0,
            confidence_score=0.8,
            voting_mode=VotingMode.WEIGHTED_VOTE,
            strategy_weights={"s1": 0.6, "s2": 0.4}
        )

        assert result.decision == EnsembleDecision.BUY
        assert result.contributing_strategies == ["s1", "s2"]
        assert result.confidence_score == 0.8
        assert result.voting_mode == VotingMode.WEIGHTED_VOTE

    def test_strategy_signal_dataclass(self):
        """Test StrategySignal dataclass creation."""
        from core.ensemble_manager import StrategySignal

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="market",
            amount=Decimal("1.0")
        )

        strategy_signal = StrategySignal(
            strategy_id="s1",
            signal=signal,
            confidence=0.8,
            weight=0.5,
            metadata={"test": "data"}
        )

        assert strategy_signal.strategy_id == "s1"
        assert strategy_signal.confidence == 0.8
        assert strategy_signal.weight == 0.5
        assert strategy_signal.metadata == {"test": "data"}


class TestEnsembleIntegration:
    """Integration tests for EnsembleManager with real strategies."""

    def test_ensemble_with_multiple_strategies(self):
        """Test ensemble manager with multiple mock strategies."""
        from strategies.base_strategy import StrategyConfig

        # Create ensemble manager
        config = {
            "enabled": True,
            "mode": "weighted_vote",
            "dynamic_weights": False,
            "strategies": [
                {"id": "rsi_strategy", "weight": 0.5},
                {"id": "ema_strategy", "weight": 0.3},
                {"id": "ml_strategy", "weight": 0.2}
            ],
            "thresholds": {
                "confidence": 0.6,
                "vote_ratio": 0.5
            }
        }
        ensemble_manager = EnsembleManager(config)

        # Create mock strategies that return different signals
        class MockRSIStrategy:
            def generate_signals(self, market_data):
                # RSI strategy: generates LONG signal when RSI < 30
                return [
                    TradingSignal(
                        strategy_id="rsi_strategy",
                        symbol="BTC/USDT",
                        signal_type=SignalType.ENTRY_LONG,
                        signal_strength=SignalStrength.STRONG,
                        order_type="market",
                        amount=Decimal("1.0"),
                        metadata={"confidence": 0.8, "rsi_value": 25}
                    )
                ]

        class MockEMAStrategy:
            def generate_signals(self, market_data):
                # EMA strategy: generates SHORT signal
                return [
                    TradingSignal(
                        strategy_id="ema_strategy",
                        symbol="BTC/USDT",
                        signal_type=SignalType.ENTRY_SHORT,
                        signal_strength=SignalStrength.MODERATE,
                        order_type="market",
                        amount=Decimal("1.0"),
                        metadata={"confidence": 0.7, "ema_signal": "bearish"}
                    )
                ]

        class MockMLStrategy:
            def generate_signals(self, market_data):
                # ML strategy: generates LONG signal with high confidence
                return [
                    TradingSignal(
                        strategy_id="ml_strategy",
                        symbol="BTC/USDT",
                        signal_type=SignalType.ENTRY_LONG,
                        signal_strength=SignalStrength.STRONG,
                        order_type="market",
                        amount=Decimal("1.0"),
                        metadata={"confidence": 0.9, "ml_prediction": 0.85}
                    )
                ]

        # Register strategies
        ensemble_manager.register_strategy("rsi_strategy", MockRSIStrategy())
        ensemble_manager.register_strategy("ema_strategy", MockEMAStrategy())
        ensemble_manager.register_strategy("ml_strategy", MockMLStrategy())

        # Test ensemble signal generation
        market_data = {"ohlcv": pd.DataFrame({"close": [50000]})}
        ensemble_signal = ensemble_manager.get_ensemble_signal(market_data)

        # Should generate a signal since majority (2 out of 3) agree on direction
        assert ensemble_signal is not None
        assert ensemble_signal.strategy_id == "ensemble"
        assert ensemble_signal.signal_type == SignalType.ENTRY_LONG  # Majority vote wins
        assert "ensemble" in ensemble_signal.metadata
        assert ensemble_signal.metadata["voting_mode"] == "weighted_vote"
        # Only RSI and ML strategies contribute to the LONG signal
        assert len(ensemble_signal.metadata["contributing_strategies"]) == 2
        assert "rsi_strategy" in ensemble_signal.metadata["contributing_strategies"]
        assert "ml_strategy" in ensemble_signal.metadata["contributing_strategies"]

    def test_ensemble_with_no_consensus(self):
        """Test ensemble manager when strategies disagree."""
        # Create ensemble manager with strict consensus requirements
        config = {
            "enabled": True,
            "mode": "majority_vote",
            "dynamic_weights": False,
            "strategies": [
                {"id": "strategy1", "weight": 1.0},
                {"id": "strategy2", "weight": 1.0},
                {"id": "strategy3", "weight": 1.0}
            ],
            "thresholds": {
                "confidence": 0.6,
                "vote_ratio": 0.8  # Require 80% consensus
            }
        }
        ensemble_manager = EnsembleManager(config)

        # Create strategies with conflicting signals
        class MockStrategy1:
            def generate_signals(self, market_data):
                return [TradingSignal(
                    strategy_id="strategy1",
                    symbol="BTC/USDT",
                    signal_type=SignalType.ENTRY_LONG,
                    signal_strength=SignalStrength.STRONG,
                    order_type="market",
                    amount=Decimal("1.0"),
                    metadata={"confidence": 0.8}
                )]

        class MockStrategy2:
            def generate_signals(self, market_data):
                return [TradingSignal(
                    strategy_id="strategy2",
                    symbol="BTC/USDT",
                    signal_type=SignalType.ENTRY_SHORT,
                    signal_strength=SignalStrength.STRONG,
                    order_type="market",
                    amount=Decimal("1.0"),
                    metadata={"confidence": 0.8}
                )]

        class MockStrategy3:
            def generate_signals(self, market_data):
                return [TradingSignal(
                    strategy_id="strategy3",
                    symbol="BTC/USDT",
                    signal_type=SignalType.EXIT_LONG,  # Exit signal
                    signal_strength=SignalStrength.STRONG,
                    order_type="market",
                    amount=Decimal("1.0"),
                    metadata={"confidence": 0.7}
                )]

        # Register strategies
        ensemble_manager.register_strategy("strategy1", MockStrategy1())
        ensemble_manager.register_strategy("strategy2", MockStrategy2())
        ensemble_manager.register_strategy("strategy3", MockStrategy3())

        # Test ensemble signal generation
        market_data = {"ohlcv": pd.DataFrame({"close": [50000]})}
        ensemble_signal = ensemble_manager.get_ensemble_signal(market_data)

        # Should not generate signal due to lack of consensus
        assert ensemble_signal is None

    def test_ensemble_disabled_fallback(self):
        """Test that individual signals pass through when ensemble is disabled."""
        config = {"enabled": False}
        ensemble_manager = EnsembleManager(config)

        # Even with strategies registered, should return None when disabled
        result = ensemble_manager.get_ensemble_signal({})
        assert result is None

    def test_ensemble_with_empty_signals(self):
        """Test ensemble manager when strategies return no signals."""
        config = {
            "enabled": True,
            "mode": "weighted_vote",
            "strategies": [{"id": "test_strategy", "weight": 1.0}],
            "thresholds": {"confidence": 0.6, "vote_ratio": 0.5}
        }
        ensemble_manager = EnsembleManager(config)

        class MockEmptyStrategy:
            def generate_signals(self, market_data):
                return []  # No signals

        ensemble_manager.register_strategy("test_strategy", MockEmptyStrategy())

        result = ensemble_manager.get_ensemble_signal({})
        assert result is None

    def test_ensemble_weight_updates(self):
        """Test that ensemble weights are updated based on performance."""
        config = {
            "enabled": True,
            "mode": "weighted_vote",
            "dynamic_weights": True,
            "strategies": [
                {"id": "good_strategy", "weight": 0.5},
                {"id": "bad_strategy", "weight": 0.5}
            ],
            "thresholds": {"confidence": 0.6, "vote_ratio": 0.5}
        }
        ensemble_manager = EnsembleManager(config)

        # Simulate performance metrics
        performance_metrics = {
            "good_strategy": {
                "sharpe_ratio": 2.0,
                "win_rate": 0.8,
                "profit_factor": 1.8
            },
            "bad_strategy": {
                "sharpe_ratio": -1.0,
                "win_rate": 0.3,
                "profit_factor": 0.7
            }
        }

        # Update weights
        ensemble_manager.update_weights(performance_metrics)

        # Good strategy should have higher weight
        assert ensemble_manager.strategy_weights["good_strategy"] > ensemble_manager.strategy_weights["bad_strategy"]
        assert ensemble_manager.strategy_weights["good_strategy"] > 0.5  # Increased
        assert ensemble_manager.strategy_weights["bad_strategy"] < 0.5  # Decreased
