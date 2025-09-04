"""
Ensemble Manager for combining multiple trading strategies.

Provides ensemble decision-making with different voting mechanisms:
- majority_vote: majority consensus required
- weighted_vote: performance-based weights
- confidence_average: ML/indicator confidence averaging

Supports dynamic weight updates based on backtest/live performance metrics.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from core.contracts import TradingSignal, SignalType, SignalStrength
from utils.config_loader import get_config
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class VotingMode(Enum):
    """Voting mechanisms for ensemble decisions."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CONFIDENCE_AVERAGE = "confidence_average"


class EnsembleDecision(Enum):
    """Possible ensemble decisions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NO_CONSENSUS = "no_consensus"


@dataclass
class StrategySignal:
    """Represents a signal from a single strategy."""
    strategy_id: str
    signal: Optional[TradingSignal]
    confidence: float = 0.5
    weight: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnsembleResult:
    """Result of ensemble decision-making."""
    decision: EnsembleDecision
    final_signal: Optional[TradingSignal]
    contributing_strategies: List[str]
    vote_counts: Dict[str, int]
    total_weight: float
    confidence_score: float
    voting_mode: VotingMode
    strategy_weights: Dict[str, float]


class EnsembleManager:
    """
    Manages ensemble of trading strategies with configurable voting mechanisms.

    Supports multiple strategies running in parallel, combining their signals
    using different voting methods, and dynamically updating weights based
    on performance metrics.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the EnsembleManager with configuration."""
        self.config = config or get_config("ensemble", {})
        self.enabled = self.config.get("enabled", False)
        self.voting_mode = VotingMode(self.config.get("mode", "weighted_vote"))
        self.dynamic_weights = self.config.get("dynamic_weights", True)
        self.thresholds = self.config.get("thresholds", {
            "confidence": 0.6,
            "vote_ratio": 0.66
        })

        # Strategy registry
        self.strategies: Dict[str, Any] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}

        # Initialize from config
        self._load_from_config()

        logger.info(f"EnsembleManager initialized: mode={self.voting_mode.value}, "
                   f"dynamic_weights={self.dynamic_weights}, enabled={self.enabled}")

    def _load_from_config(self) -> None:
        """Load strategy configurations from config."""
        strategies_config = self.config.get("strategies", [])
        for strategy_config in strategies_config:
            strategy_id = strategy_config.get("id")
            weight = strategy_config.get("weight", 1.0)
            if strategy_id:
                self.strategy_weights[strategy_id] = weight
                # Initialize empty performance metrics
                self.performance_metrics[strategy_id] = {
                    "sharpe_ratio": 0.0,
                    "win_rate": 0.5,
                    "expectancy": 0.0,
                    "profit_factor": 1.0,
                    "total_trades": 0
                }

    def register_strategy(self, strategy_id: str, strategy_object: Any) -> None:
        """
        Register a strategy for ensemble use.

        Args:
            strategy_id: Unique identifier for the strategy
            strategy_object: Strategy instance with generate_signals method
        """
        self.strategies[strategy_id] = strategy_object

        # Initialize weight if not set
        if strategy_id not in self.strategy_weights:
            self.strategy_weights[strategy_id] = 1.0

        # Initialize performance metrics if not present
        if strategy_id not in self.performance_metrics:
            self.performance_metrics[strategy_id] = {
                "sharpe_ratio": 0.0,
                "win_rate": 0.5,
                "expectancy": 0.0,
                "profit_factor": 1.0,
                "total_trades": 0
            }

        logger.info(f"Strategy registered: {strategy_id}")

    def get_ensemble_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Generate ensemble signal from all registered strategies.

        Args:
            market_data: Market data dictionary

        Returns:
            Ensemble trading signal or None if no consensus
        """
        if not self.enabled or not self.strategies:
            return None

        # Collect signals from all strategies
        strategy_signals = []
        for strategy_id, strategy in self.strategies.items():
            try:
                # Generate signal from strategy
                signals = strategy.generate_signals(market_data)
                if signals:
                    # Take the first/primary signal
                    signal = signals[0]
                    confidence = self._extract_confidence(signal, market_data)
                    weight = self.strategy_weights.get(strategy_id, 1.0)

                    strategy_signal = StrategySignal(
                        strategy_id=strategy_id,
                        signal=signal,
                        confidence=confidence,
                        weight=weight,
                        metadata=getattr(signal, 'metadata', {})
                    )
                    strategy_signals.append(strategy_signal)
            except Exception as e:
                logger.warning(f"Error generating signal from {strategy_id}: {e}")
                continue

        if not strategy_signals:
            return None

        # Make ensemble decision
        result = self._make_ensemble_decision(strategy_signals)

        # Log ensemble decision
        self._log_ensemble_decision(result)

        return result.final_signal

    def _extract_confidence(self, signal: TradingSignal, market_data: Dict[str, Any]) -> float:
        """
        Extract confidence score from signal metadata or market data.

        Args:
            signal: Trading signal
            market_data: Market data

        Returns:
            Confidence score between 0 and 1
        """
        # Check market data for ML predictions first (higher priority)
        if 'ml_prediction' in market_data:
            pred = market_data['ml_prediction']
            if isinstance(pred, dict) and 'confidence' in pred:
                return float(pred['confidence'])

        # Check signal metadata for confidence
        if hasattr(signal, 'metadata') and signal.metadata:
            confidence = signal.metadata.get('confidence', 0.5)
            if isinstance(confidence, (int, float)):
                return float(confidence)

        # Default confidence based on signal strength
        strength_confidence = {
            SignalStrength.WEAK: 0.3,
            SignalStrength.MODERATE: 0.6,
            SignalStrength.STRONG: 0.9,
            SignalStrength.EXTREME: 1.0
        }
        return strength_confidence.get(signal.signal_strength, 0.5)

    def _make_ensemble_decision(self, strategy_signals: List[StrategySignal]) -> EnsembleResult:
        """
        Make ensemble decision based on voting mode.

        Args:
            strategy_signals: List of strategy signals

        Returns:
            EnsembleResult with decision and metadata
        """
        if self.voting_mode == VotingMode.MAJORITY_VOTE:
            return self._majority_vote(strategy_signals)
        elif self.voting_mode == VotingMode.WEIGHTED_VOTE:
            return self._weighted_vote(strategy_signals)
        elif self.voting_mode == VotingMode.CONFIDENCE_AVERAGE:
            return self._confidence_average(strategy_signals)
        else:
            logger.error(f"Unknown voting mode: {self.voting_mode}")
            return EnsembleResult(
                decision=EnsembleDecision.NO_CONSENSUS,
                final_signal=None,
                contributing_strategies=[],
                vote_counts={},
                total_weight=0.0,
                confidence_score=0.0,
                voting_mode=self.voting_mode,
                strategy_weights={}
            )

    def _majority_vote(self, strategy_signals: List[StrategySignal]) -> EnsembleResult:
        """Implement majority voting mechanism."""
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        total_weight = sum(s.weight for s in strategy_signals)

        contributing_strategies = []
        vote_counts = {"buy": 0, "sell": 0, "hold": 0}

        for sig in strategy_signals:
            if sig.signal:
                if sig.signal.signal_type == SignalType.ENTRY_LONG:
                    buy_votes += sig.weight
                    vote_counts["buy"] += 1
                elif sig.signal.signal_type == SignalType.ENTRY_SHORT:
                    sell_votes += sig.weight
                    vote_counts["sell"] += 1
                else:
                    hold_votes += sig.weight
                    vote_counts["hold"] += 1
                contributing_strategies.append(sig.strategy_id)

        # Check majority threshold
        max_votes = max(buy_votes, sell_votes, hold_votes)
        threshold = total_weight * self.thresholds["vote_ratio"]

        if max_votes >= threshold:
            if buy_votes == max_votes:
                decision = EnsembleDecision.BUY
                final_signal = self._create_ensemble_signal(strategy_signals, SignalType.ENTRY_LONG)
            elif sell_votes == max_votes:
                decision = EnsembleDecision.SELL
                final_signal = self._create_ensemble_signal(strategy_signals, SignalType.ENTRY_SHORT)
            else:
                decision = EnsembleDecision.HOLD
                final_signal = None
        else:
            decision = EnsembleDecision.NO_CONSENSUS
            final_signal = None

        confidence_score = max_votes / total_weight if total_weight > 0 else 0.0

        return EnsembleResult(
            decision=decision,
            final_signal=final_signal,
            contributing_strategies=contributing_strategies,
            vote_counts=vote_counts,
            total_weight=total_weight,
            confidence_score=confidence_score,
            voting_mode=VotingMode.MAJORITY_VOTE,
            strategy_weights={s.strategy_id: s.weight for s in strategy_signals}
        )

    def _weighted_vote(self, strategy_signals: List[StrategySignal]) -> EnsembleResult:
        """Implement weighted voting based on strategy performance."""
        buy_weight = 0.0
        sell_weight = 0.0
        hold_weight = 0.0
        total_weight = sum(s.weight for s in strategy_signals)

        contributing_strategies = []
        vote_counts = {"buy": 0, "sell": 0, "hold": 0}

        for sig in strategy_signals:
            if sig.signal:
                weight = sig.weight
                if sig.signal.signal_type == SignalType.ENTRY_LONG:
                    buy_weight += weight
                    vote_counts["buy"] += 1
                elif sig.signal.signal_type == SignalType.ENTRY_SHORT:
                    sell_weight += weight
                    vote_counts["sell"] += 1
                else:
                    hold_weight += weight
                    vote_counts["hold"] += 1
                contributing_strategies.append(sig.strategy_id)

        # Find winning direction
        weights = {"buy": buy_weight, "sell": sell_weight, "hold": hold_weight}
        max_direction = max(weights, key=weights.get)
        max_weight = weights[max_direction]

        if max_weight > total_weight * self.thresholds["vote_ratio"]:
            if max_direction == "buy":
                decision = EnsembleDecision.BUY
                final_signal = self._create_ensemble_signal(strategy_signals, SignalType.ENTRY_LONG)
            elif max_direction == "sell":
                decision = EnsembleDecision.SELL
                final_signal = self._create_ensemble_signal(strategy_signals, SignalType.ENTRY_SHORT)
            else:
                decision = EnsembleDecision.HOLD
                final_signal = None
        else:
            decision = EnsembleDecision.NO_CONSENSUS
            final_signal = None

        confidence_score = max_weight / total_weight if total_weight > 0 else 0.0

        return EnsembleResult(
            decision=decision,
            final_signal=final_signal,
            contributing_strategies=contributing_strategies,
            vote_counts=vote_counts,
            total_weight=total_weight,
            confidence_score=confidence_score,
            voting_mode=VotingMode.WEIGHTED_VOTE,
            strategy_weights={s.strategy_id: s.weight for s in strategy_signals}
        )

    def _confidence_average(self, strategy_signals: List[StrategySignal]) -> EnsembleResult:
        """Implement confidence averaging mechanism."""
        buy_confidences = []
        sell_confidences = []
        hold_confidences = []

        contributing_strategies = []
        vote_counts = {"buy": 0, "sell": 0, "hold": 0}
        total_weight = sum(s.weight for s in strategy_signals)

        for sig in strategy_signals:
            if sig.signal:
                confidence = sig.confidence
                if sig.signal.signal_type == SignalType.ENTRY_LONG:
                    buy_confidences.append(confidence)
                    vote_counts["buy"] += 1
                elif sig.signal.signal_type == SignalType.ENTRY_SHORT:
                    sell_confidences.append(confidence)
                    vote_counts["sell"] += 1
                else:
                    hold_confidences.append(confidence)
                    vote_counts["hold"] += 1
                contributing_strategies.append(sig.strategy_id)

        # Calculate average confidences
        avg_buy = np.mean(buy_confidences) if buy_confidences else 0.0
        avg_sell = np.mean(sell_confidences) if sell_confidences else 0.0
        avg_hold = np.mean(hold_confidences) if hold_confidences else 0.0

        # Check if we have any entry signals (buy or sell)
        has_entry_signals = (buy_confidences or sell_confidences)

        if not has_entry_signals:
            # No entry signals, cannot make a trading decision
            return EnsembleResult(
                decision=EnsembleDecision.NO_CONSENSUS,
                final_signal=None,
                contributing_strategies=contributing_strategies,
                vote_counts=vote_counts,
                total_weight=total_weight,
                confidence_score=0.0,
                voting_mode=VotingMode.CONFIDENCE_AVERAGE,
                strategy_weights={s.strategy_id: s.weight for s in strategy_signals}
            )

        # Find direction with highest average confidence
        confidences = {"buy": avg_buy, "sell": avg_sell, "hold": avg_hold}
        max_direction = max(confidences, key=confidences.get)
        max_confidence = confidences[max_direction]

        if max_confidence >= self.thresholds["confidence"]:
            if max_direction == "buy":
                decision = EnsembleDecision.BUY
                final_signal = self._create_ensemble_signal(strategy_signals, SignalType.ENTRY_LONG)
            elif max_direction == "sell":
                decision = EnsembleDecision.SELL
                final_signal = self._create_ensemble_signal(strategy_signals, SignalType.ENTRY_SHORT)
            else:
                decision = EnsembleDecision.HOLD
                final_signal = None
        else:
            decision = EnsembleDecision.NO_CONSENSUS
            final_signal = None

        return EnsembleResult(
            decision=decision,
            final_signal=final_signal,
            contributing_strategies=contributing_strategies,
            vote_counts=vote_counts,
            total_weight=total_weight,
            confidence_score=max_confidence,
            voting_mode=VotingMode.CONFIDENCE_AVERAGE,
            strategy_weights={s.strategy_id: s.weight for s in strategy_signals}
        )

    def _create_ensemble_signal(self, strategy_signals: List[StrategySignal],
                               signal_type: SignalType) -> TradingSignal:
        """
        Create ensemble signal by combining relevant strategy signals.

        Args:
            strategy_signals: All strategy signals
            signal_type: Desired signal type for ensemble

        Returns:
            Combined trading signal
        """
        # Filter signals matching the desired type
        relevant_signals = [
            s for s in strategy_signals
            if s.signal and s.signal.signal_type == signal_type
        ]

        if not relevant_signals:
            return None

        # Use the signal with highest confidence/weight as base
        base_signal = max(relevant_signals,
                         key=lambda s: s.confidence * s.weight).signal

        # Create ensemble metadata
        ensemble_metadata = {
            "ensemble": True,
            "contributing_strategies": [s.strategy_id for s in relevant_signals],
            "strategy_weights": {s.strategy_id: s.weight for s in relevant_signals},
            "average_confidence": np.mean([s.confidence for s in relevant_signals]),
            "voting_mode": self.voting_mode.value
        }

        # Merge metadata
        combined_metadata = dict(base_signal.metadata or {})
        combined_metadata.update(ensemble_metadata)

        # Create new signal with ensemble metadata
        ensemble_signal = TradingSignal(
            strategy_id="ensemble",
            symbol=base_signal.symbol,
            signal_type=signal_type,
            signal_strength=base_signal.signal_strength,
            order_type=base_signal.order_type,
            amount=base_signal.amount,
            price=base_signal.price,
            current_price=base_signal.current_price,
            stop_loss=base_signal.stop_loss,
            take_profit=base_signal.take_profit,
            trailing_stop=base_signal.trailing_stop,
            timestamp=base_signal.timestamp,
            metadata=combined_metadata
        )

        return ensemble_signal

    def update_weights(self, performance_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Update strategy weights based on performance metrics.

        Args:
            performance_metrics: Dict of strategy_id -> performance metrics
        """
        if not self.dynamic_weights:
            return

        self.performance_metrics.update(performance_metrics)

        # Calculate new weights based on performance
        for strategy_id in self.strategy_weights:
            metrics = self.performance_metrics.get(strategy_id, {})
            new_weight = self._calculate_weight_from_metrics(metrics)
            self.strategy_weights[strategy_id] = new_weight

        logger.info(f"Updated strategy weights: {self.strategy_weights}")

    def _calculate_weight_from_metrics(self, metrics: Dict[str, float]) -> float:
        """
        Calculate strategy weight from performance metrics.

        Args:
            metrics: Performance metrics dictionary

        Returns:
            Calculated weight
        """
        # Use Sharpe ratio as primary weight factor
        sharpe = metrics.get("sharpe_ratio", 0.0)
        win_rate = metrics.get("win_rate", 0.5)
        profit_factor = metrics.get("profit_factor", 1.0)

        # Normalize Sharpe ratio (assuming -3 to +3 range)
        normalized_sharpe = max(0, (sharpe + 3) / 6)

        # Combine factors
        weight = (normalized_sharpe * 0.4 + win_rate * 0.4 + min(profit_factor / 2, 1.0) * 0.2)

        # Ensure minimum weight
        return max(weight, 0.1)

    def _log_ensemble_decision(self, result: EnsembleResult) -> None:
        """Log ensemble decision details."""
        if result.final_signal:
            trade_logger.trade("Ensemble Decision", {
                "decision": result.decision.value,
                "contributing_strategies": result.contributing_strategies,
                "vote_counts": result.vote_counts,
                "total_weight": result.total_weight,
                "confidence_score": result.confidence_score,
                "voting_mode": result.voting_mode.value,
                "strategy_weights": result.strategy_weights,
                "symbol": result.final_signal.symbol if result.final_signal else None
            })

    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get current performance metrics for all strategies."""
        return self.performance_metrics.copy()

    def get_strategy_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        return self.strategy_weights.copy()

    def set_voting_mode(self, mode: VotingMode) -> None:
        """Change the voting mode."""
        self.voting_mode = mode
        logger.info(f"Voting mode changed to: {mode.value}")

    def enable_ensemble(self, enabled: bool = True) -> None:
        """Enable or disable ensemble processing."""
        self.enabled = enabled
        logger.info(f"Ensemble {'enabled' if enabled else 'disabled'}")
