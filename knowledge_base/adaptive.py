"""
Adaptive Weighting Logic for Knowledge-Based Strategy Selection

This module implements algorithms for weighting strategies based on historical
performance data stored in the knowledge base. It provides adaptive mechanisms
that learn from past trading outcomes to improve future strategy selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .schema import (
    KnowledgeEntry, KnowledgeQuery, KnowledgeQueryResult,
    MarketRegime, StrategyCategory, MarketCondition, StrategyMetadata
)
from .storage import KnowledgeStorage

logger = logging.getLogger(__name__)


class AdaptiveWeightingEngine:
    """
    Engine for calculating adaptive weights based on historical knowledge.

    This class implements various weighting algorithms that consider:
    - Historical performance metrics
    - Market regime similarity
    - Strategy category effectiveness
    - Recency and sample size factors
    - Risk-adjusted returns
    """

    def __init__(self, storage: KnowledgeStorage, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adaptive weighting engine.

        Args:
            storage: Knowledge storage backend
            config: Configuration parameters
        """
        self.storage = storage
        self.config = config or self._get_default_config()

        # Weighting algorithm parameters
        self.performance_weight = self.config.get('performance_weight', 0.4)
        self.regime_similarity_weight = self.config.get('regime_similarity_weight', 0.3)
        self.recency_weight = self.config.get('recency_weight', 0.2)
        self.sample_size_weight = self.config.get('sample_size_weight', 0.1)

        # Decay factors
        self.performance_decay_days = self.config.get('performance_decay_days', 90)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)

        # Caching for performance
        self._weight_cache: Dict[str, Dict[str, float]] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=5)  # Cache weights for 5 minutes

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'performance_weight': 0.4,
            'regime_similarity_weight': 0.3,
            'recency_weight': 0.2,
            'sample_size_weight': 0.1,
            'performance_decay_days': 90,
            'confidence_threshold': 0.3,
            'min_sample_size': 5,
            'max_weight': 3.0,
            'min_weight': 0.1
        }

    def calculate_adaptive_weights(
        self,
        current_market: MarketCondition,
        available_strategies: List[StrategyMetadata],
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights for strategies based on historical knowledge.

        Args:
            current_market: Current market conditions
            available_strategies: List of available strategy metadata
            base_weights: Optional base weights to adjust

        Returns:
            Dictionary mapping strategy names to adaptive weights
        """
        # Check cache first
        cache_key = self._get_cache_key(current_market, available_strategies)
        if self._is_cache_valid():
            cached_weights = self._weight_cache.get(cache_key)
            if cached_weights:
                return cached_weights

        # Initialize weights
        adaptive_weights = base_weights.copy() if base_weights else {}
        strategy_names = [s.name for s in available_strategies]

        for strategy_name in strategy_names:
            if strategy_name not in adaptive_weights:
                adaptive_weights[strategy_name] = 1.0

        # Query relevant knowledge entries
        relevant_knowledge = self._query_relevant_knowledge(current_market, available_strategies)

        if not relevant_knowledge.entries:
            logger.info("No relevant knowledge found, using base weights")
            return adaptive_weights

        # Calculate weights for each strategy
        for strategy_meta in available_strategies:
            strategy_name = strategy_meta.name
            strategy_knowledge = self._filter_strategy_knowledge(relevant_knowledge, strategy_name)

            if strategy_knowledge:
                adaptive_weight = self._calculate_strategy_weight(
                    strategy_meta, strategy_knowledge, current_market
                )
                adaptive_weights[strategy_name] = self._clamp_weight(adaptive_weight)
            else:
                # Reduce weight for strategies with no historical knowledge
                adaptive_weights[strategy_name] *= 0.8

        # Normalize weights to maintain relative relationships
        adaptive_weights = self._normalize_weights(adaptive_weights)

        # Cache the results
        self._weight_cache[cache_key] = adaptive_weights
        self._cache_timestamp = datetime.now()

        logger.info(f"Calculated adaptive weights: {adaptive_weights}")
        return adaptive_weights

    def _query_relevant_knowledge(
        self,
        market_condition: MarketCondition,
        strategies: List[StrategyMetadata]
    ) -> KnowledgeQueryResult:
        """Query knowledge base for relevant historical data."""
        # Create query for similar market conditions
        query = KnowledgeQuery(
            market_regime=market_condition.regime,
            min_confidence=self.confidence_threshold,
            min_sample_size=self.config.get('min_sample_size', 5),
            limit=50  # Get more data for better analysis
        )

        # Also query for strategies in the same category
        strategy_categories = list(set(s.category for s in strategies))
        if len(strategy_categories) == 1:
            query.strategy_category = strategy_categories[0]

        return self.storage.query_entries(query)

    def _filter_strategy_knowledge(
        self,
        knowledge_result: KnowledgeQueryResult,
        strategy_name: str
    ) -> List[KnowledgeEntry]:
        """Filter knowledge entries for a specific strategy."""
        return [
            entry for entry in knowledge_result.entries
            if entry.strategy_metadata.name == strategy_name
        ]

    def _calculate_strategy_weight(
        self,
        strategy_meta: StrategyMetadata,
        knowledge_entries: List[KnowledgeEntry],
        current_market: MarketCondition
    ) -> float:
        """Calculate adaptive weight for a strategy based on its knowledge."""
        if not knowledge_entries:
            return 1.0

        # Aggregate performance metrics across all relevant entries
        total_weighted_score = 0.0
        total_weight = 0.0

        for entry in knowledge_entries:
            # Calculate relevance weight based on market similarity
            market_similarity = self._calculate_market_similarity(
                current_market, entry.market_condition
            )

            # Calculate recency weight
            recency_weight = self._calculate_recency_weight(entry.last_updated)

            # Calculate sample size weight
            sample_weight = min(1.0, entry.sample_size / 20)  # Cap at 20 samples

            # Combine weights
            combined_weight = (
                market_similarity * self.regime_similarity_weight +
                recency_weight * self.recency_weight +
                sample_weight * self.sample_size_weight
            )

            # Performance score based on multiple metrics
            performance_score = self._calculate_performance_score(entry)

            # Apply combined weight to performance score
            weighted_score = performance_score * combined_weight
            total_weighted_score += weighted_score
            total_weight += combined_weight

        if total_weight == 0:
            return 1.0

        # Average weighted score
        avg_score = total_weighted_score / total_weight

        # Convert score to weight multiplier (1.0 = neutral, >1.0 = boost, <1.0 = reduce)
        weight_multiplier = 1.0 + (avg_score - 0.5) * 0.5  # Scale factor

        return weight_multiplier

    def _calculate_market_similarity(
        self,
        current_market: MarketCondition,
        historical_market: MarketCondition
    ) -> float:
        """Calculate similarity between current and historical market conditions."""
        similarity_score = 0.0
        factors = 0

        # Regime match (exact match gets full weight)
        if current_market.regime == historical_market.regime:
            similarity_score += 1.0
        factors += 1

        # Volatility similarity (closer values get higher similarity)
        if current_market.volatility > 0 and historical_market.volatility > 0:
            vol_ratio = min(current_market.volatility, historical_market.volatility) / \
                       max(current_market.volatility, historical_market.volatility)
            similarity_score += vol_ratio
            factors += 1

        # Trend strength similarity
        if current_market.trend_strength > 0 and historical_market.trend_strength > 0:
            trend_ratio = min(current_market.trend_strength, historical_market.trend_strength) / \
                         max(current_market.trend_strength, historical_market.trend_strength)
            similarity_score += trend_ratio
            factors += 1

        return similarity_score / factors if factors > 0 else 0.5

    def _calculate_recency_weight(self, entry_timestamp: datetime) -> float:
        """Calculate weight based on how recent the knowledge entry is."""
        days_old = (datetime.now() - entry_timestamp).days

        if days_old <= 7:
            return 1.0  # Very recent
        elif days_old <= 30:
            return 0.8  # Recent
        elif days_old <= self.performance_decay_days:
            # Linear decay from 0.8 to 0.3
            decay_factor = 0.8 - (0.5 * (days_old - 30) / (self.performance_decay_days - 30))
            return max(0.3, decay_factor)
        else:
            return 0.3  # Old knowledge gets reduced weight

    def _calculate_performance_score(self, perf: 'PerformanceMetrics') -> float:
        """Calculate a composite performance score from multiple metrics."""

        # Normalize each metric to 0-1 scale
        win_rate_score = perf.win_rate
        profit_factor_score = min(1.0, perf.profit_factor / 2.0)  # Cap at profit factor of 2
        sharpe_score = min(1.0, max(0.0, (perf.sharpe_ratio + 2) / 4))  # Normalize around 0

        # Risk-adjusted score (lower drawdown is better)
        risk_score = max(0.0, 1.0 - (perf.max_drawdown / 0.5))  # Assume 50% max acceptable drawdown

        # Weighted combination
        composite_score = (
            win_rate_score * 0.3 +
            profit_factor_score * 0.3 +
            sharpe_score * 0.2 +
            risk_score * 0.2
        )

        return composite_score

    def _clamp_weight(self, weight: float) -> float:
        """Clamp weight to configured min/max values."""
        return max(
            self.config.get('min_weight', 0.1),
            min(self.config.get('max_weight', 3.0), weight)
        )

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to maintain relative relationships."""
        if not weights:
            return weights

        # Calculate geometric mean as normalization factor
        values = list(weights.values())
        if len(values) > 1:
            geo_mean = np.exp(np.mean(np.log(values)))
            # Scale weights so geometric mean becomes 1.0
            scale_factor = 1.0 / geo_mean
            normalized = {k: v * scale_factor for k, v in weights.items()}
        else:
            normalized = weights.copy()

        return normalized

    def _get_cache_key(
        self,
        market_condition: MarketCondition,
        strategies: List[StrategyMetadata]
    ) -> str:
        """Generate cache key for weight calculations."""
        strategy_names = sorted([s.name for s in strategies])
        regime = market_condition.regime.value
        return f"{regime}_{'_'.join(strategy_names)}_{market_condition.timestamp.date()}"

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_timestamp is None:
            return False
        return (datetime.now() - self._cache_timestamp) < self._cache_ttl

    def get_strategy_recommendations(
        self,
        current_market: MarketCondition,
        available_strategies: List[StrategyMetadata],
        top_n: int = 3
    ) -> List[Tuple[str, float, str]]:
        """
        Get top strategy recommendations with weights and reasoning.

        Args:
            current_market: Current market conditions
            available_strategies: Available strategies
            top_n: Number of top recommendations to return

        Returns:
            List of (strategy_name, weight, reasoning) tuples
        """
        weights = self.calculate_adaptive_weights(current_market, available_strategies)

        # Sort strategies by weight
        sorted_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        recommendations = []
        for strategy_name, weight in sorted_strategies[:top_n]:
            reasoning = self._generate_recommendation_reasoning(
                strategy_name, weight, current_market
            )
            recommendations.append((strategy_name, weight, reasoning))

        return recommendations

    def _generate_recommendation_reasoning(
        self,
        strategy_name: str,
        weight: float,
        market_condition: MarketCondition
    ) -> str:
        """Generate human-readable reasoning for a recommendation."""
        if weight > 1.5:
            confidence = "High confidence"
        elif weight > 1.2:
            confidence = "Moderate confidence"
        elif weight > 0.8:
            confidence = "Low confidence"
        else:
            confidence = "Very low confidence"

        regime_name = market_condition.regime.value.replace('_', ' ').title()

        return f"{confidence} recommendation for {strategy_name} in {regime_name} conditions (weight: {weight:.2f})"

    def update_knowledge_from_trade(
        self,
        strategy_name: str,
        market_condition: MarketCondition,
        trade_result: Dict[str, Any]
    ) -> bool:
        """
        Update knowledge base with results from a completed trade.

        Args:
            strategy_name: Name of the strategy used
            market_condition: Market conditions during the trade
            trade_result: Trade performance metrics

        Returns:
            Success status
        """
        try:
            # Create or update knowledge entry
            entry_id = self._generate_entry_id(strategy_name, market_condition)

            # Check if entry already exists
            existing_entry = self.storage.get_entry(entry_id)

            if existing_entry:
                # Update existing entry
                new_performance = self._extract_performance_from_trade(trade_result)
                existing_entry.update_performance(new_performance)
                success = self.storage.save_entry(existing_entry)
                action = "updated"
            else:
                # Create new entry
                strategy_meta = self._create_strategy_metadata(strategy_name)
                performance = self._extract_performance_from_trade(trade_result)
                outcome = self._determine_trade_outcome(trade_result)

                new_entry = KnowledgeEntry(
                    id=entry_id,
                    market_condition=market_condition,
                    strategy_metadata=strategy_meta,
                    performance=performance,
                    outcome=outcome,
                    confidence_score=0.5,  # Initial confidence
                    sample_size=1,
                    last_updated=datetime.now()
                )

                success = self.storage.save_entry(new_entry)
                action = "created"

            if success:
                logger.info(f"Knowledge entry {action} for strategy {strategy_name}")
                # Invalidate cache since knowledge has changed
                self._weight_cache.clear()
                self._cache_timestamp = None

            return success

        except Exception as e:
            logger.error(f"Failed to update knowledge from trade: {e}")
            return False

    def _generate_entry_id(self, strategy_name: str, market_condition: MarketCondition) -> str:
        """Generate unique ID for knowledge entry."""
        import hashlib
        key = f"{strategy_name}_{market_condition.regime.value}_{market_condition.timestamp.date()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _extract_performance_from_trade(self, trade_result: Dict[str, Any]) -> 'PerformanceMetrics':
        """Extract performance metrics from trade result."""
        from .schema import PerformanceMetrics

        return PerformanceMetrics(
            total_trades=1,
            winning_trades=1 if trade_result.get('pnl', 0) > 0 else 0,
            losing_trades=1 if trade_result.get('pnl', 0) <= 0 else 0,
            win_rate=1.0 if trade_result.get('pnl', 0) > 0 else 0.0,
            profit_factor=max(0.1, abs(trade_result.get('pnl', 0)) / max(0.01, abs(trade_result.get('entry_price', 1)))),
            sharpe_ratio=trade_result.get('sharpe_ratio', 0.0),
            max_drawdown=trade_result.get('max_drawdown', 0.0),
            avg_win=trade_result.get('pnl', 0) if trade_result.get('pnl', 0) > 0 else 0.0,
            avg_loss=abs(trade_result.get('pnl', 0)) if trade_result.get('pnl', 0) <= 0 else 0.0,
            total_pnl=trade_result.get('pnl', 0.0),
            total_returns=trade_result.get('returns', 0.0)
        )

    def _create_strategy_metadata(self, strategy_name: str) -> StrategyMetadata:
        """Create strategy metadata (simplified version)."""
        from .schema import StrategyMetadata, StrategyCategory

        # This is a simplified version - in practice, you'd look up the actual strategy
        return StrategyMetadata(
            name=strategy_name,
            category=StrategyCategory.TREND_FOLLOWING,  # Default
            parameters={},
            timeframe="1h",
            indicators_used=["price"],
            risk_profile="medium"
        )

    def _determine_trade_outcome(self, trade_result: Dict[str, Any]) -> 'OutcomeTag':
        """Determine the outcome tag for a trade."""
        from .schema import OutcomeTag

        pnl = trade_result.get('pnl', 0)
        if pnl > 0:
            return OutcomeTag.SUCCESS
        elif pnl < 0:
            return OutcomeTag.FAILURE
        else:
            return OutcomeTag.BREAK_EVEN

    def get_adaptive_statistics(self) -> Dict[str, Any]:
        """Get statistics about the adaptive learning system."""
        stats = self.storage.get_stats()

        # Add adaptive-specific stats
        stats.update({
            'cache_size': len(self._weight_cache),
            'cache_age_minutes': (
                (datetime.now() - self._cache_timestamp).total_seconds() / 60
                if self._cache_timestamp else None
            ),
            'performance_weight': self.performance_weight,
            'regime_similarity_weight': self.regime_similarity_weight,
            'recency_weight': self.recency_weight,
            'sample_size_weight': self.sample_size_weight
        })

        return stats
