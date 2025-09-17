"""
Adaptive Weighting Logic for Knowledge-Based Strategy Selection

This module implements algorithms for weighting strategies based on historical
performance data stored in the knowledge base. It provides adaptive mechanisms
that learn from past trading outcomes to improve future strategy selection.

Naming conventions have been standardized to follow PEP 8 (snake_case for variables and methods).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .schema import (
    KnowledgeEntry, KnowledgeQuery, KnowledgeQueryResult,
    MarketRegime, StrategyCategory, MarketCondition, StrategyMetadata
)
from .storage import KnowledgeStorage

logger = logging.getLogger(__name__)

# Default configuration constants for WeightingCalculator
PERFORMANCE_WEIGHT_DEFAULT = 0.4
REGIME_SIMILARITY_WEIGHT_DEFAULT = 0.3
RECENCY_WEIGHT_DEFAULT = 0.2
SAMPLE_SIZE_WEIGHT_DEFAULT = 0.1
PERFORMANCE_DECAY_DAYS_DEFAULT = 90
CONFIDENCE_THRESHOLD_DEFAULT = 0.3
MIN_WEIGHT_DEFAULT = 0.1
MAX_WEIGHT_DEFAULT = 3.0

# Constants for recency weight calculation
RECENCY_VERY_RECENT_DAYS = 7
RECENCY_RECENT_DAYS = 30
RECENCY_MIN_WEIGHT = 0.3
RECENCY_MAX_WEIGHT = 1.0
RECENCY_DECAY_START = 0.8

# Constants for sample size weight calculation
SAMPLE_SIZE_LOG_BASE = 101

# Constants for performance score calculation
PERF_WIN_RATE_WEIGHT = 0.3
PERF_PROFIT_FACTOR_WEIGHT = 0.3
PERF_SHARPE_WEIGHT = 0.2
PERF_RISK_WEIGHT = 0.2
PROFIT_FACTOR_CAP = 2.0
SHARPE_NORMALIZE_MIN = -2
SHARPE_NORMALIZE_MAX = 2
MAX_DRAWDOWN_ACCEPTABLE = 0.5
RISK_SCORE_DEFAULT = 0.5

# Constants for strategy weight calculation
NO_KNOWLEDGE_DEFAULT_WEIGHT = 1.0
WEIGHT_MULTIPLIER_SCALE = 0.5
SCORE_NEUTRAL = 0.5
NO_STRATEGY_KNOWLEDGE_MULTIPLIER = 0.8

# Constants for weight normalization
NORMALIZATION_TOLERANCE = 1e-6

# Constants for CacheManager
CACHE_TTL_MINUTES_DEFAULT = 5

# Constants for AdaptiveWeightingEngine
MIN_SAMPLE_SIZE_DEFAULT = 5
QUERY_LIMIT = 50
TOP_N_DEFAULT = 3
CONFIDENCE_HIGH = 1.5
CONFIDENCE_MODERATE = 1.2
CONFIDENCE_LOW = 0.8
INITIAL_CONFIDENCE = 0.5
PROFIT_FACTOR_MIN = 0.1
ENTRY_PRICE_MIN = 0.01


class WeightingCalculator:
    """
    Handles core weight calculation logic for adaptive strategy weighting.

    This class focuses on the mathematical computations involved in calculating
    adaptive weights based on historical performance data and market conditions.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the weighting calculator.

        Args:
            config: Configuration parameters for weight calculations
        """
        self.performance_weight = config.get('performance_weight', PERFORMANCE_WEIGHT_DEFAULT)
        self.regime_similarity_weight = config.get('regime_similarity_weight', REGIME_SIMILARITY_WEIGHT_DEFAULT)
        self.recency_weight = config.get('recency_weight', RECENCY_WEIGHT_DEFAULT)
        self.sample_size_weight = config.get('sample_size_weight', SAMPLE_SIZE_WEIGHT_DEFAULT)
        self.performance_decay_days = config.get('performance_decay_days', PERFORMANCE_DECAY_DAYS_DEFAULT)
        self.confidence_threshold = config.get('confidence_threshold', CONFIDENCE_THRESHOLD_DEFAULT)
        self.min_weight = config.get('min_weight', MIN_WEIGHT_DEFAULT)
        self.max_weight = config.get('max_weight', MAX_WEIGHT_DEFAULT)

    def calculate_market_similarity(
        self,
        current_market: MarketCondition,
        historical_market: MarketCondition
    ) -> float:
        """
        Calculate similarity between current and historical market conditions.

        Args:
            current_market: Current market conditions
            historical_market: Historical market conditions

        Returns:
            Similarity score between 0.0 and 1.0
        """
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

    def calculate_recency_weight(self, entry_timestamp: datetime) -> float:
        """
        Calculate weight based on how recent the knowledge entry is.

        Args:
            entry_timestamp: Timestamp of the knowledge entry

        Returns:
            Recency weight between RECENCY_MIN_WEIGHT and RECENCY_MAX_WEIGHT
        """
        days_old = (datetime.now() - entry_timestamp).days

        if days_old <= RECENCY_VERY_RECENT_DAYS:
            return RECENCY_MAX_WEIGHT  # Very recent
        elif days_old <= RECENCY_RECENT_DAYS:
            return RECENCY_DECAY_START  # Recent
        elif days_old <= self.performance_decay_days:
            # Linear decay from RECENCY_DECAY_START to RECENCY_MIN_WEIGHT
            decay_factor = RECENCY_DECAY_START - (RECENCY_DECAY_START - RECENCY_MIN_WEIGHT) * (days_old - RECENCY_RECENT_DAYS) / (self.performance_decay_days - RECENCY_RECENT_DAYS)
            return max(RECENCY_MIN_WEIGHT, decay_factor)
        else:
            return RECENCY_MIN_WEIGHT  # Old knowledge gets reduced weight

    def calculate_sample_size_weight(self, sample_size: int) -> float:
        """
        Calculate weight based on sample size using logarithmic function.

        Args:
            sample_size: Number of samples in the knowledge entry

        Returns:
            Sample size weight capped at 1.0
        """
        return min(1.0, np.log(sample_size + 1) / np.log(SAMPLE_SIZE_LOG_BASE))  # Diminishing to 1.0 at ~100 samples

    def calculate_performance_score(self, perf: 'PerformanceMetrics') -> float:
        """
        Calculate a composite performance score from multiple metrics.

        Args:
            perf: Performance metrics object

        Returns:
            Composite performance score between 0.0 and 1.0
        """
        # Normalize each metric to 0-1 scale with zero/None checks to prevent division errors
        win_rate_score = perf.win_rate if perf.win_rate is not None else 0.0

        # Handle potential None or zero profit_factor to avoid division issues
        if perf.profit_factor is None or perf.profit_factor <= 0:
            profit_factor_score = 0.0  # Default to 0 if invalid, preventing system disruption
        else:
            profit_factor_score = min(1.0, perf.profit_factor / PROFIT_FACTOR_CAP)  # Cap at profit factor of PROFIT_FACTOR_CAP

        # Handle potential None sharpe_ratio
        sharpe_val = perf.sharpe_ratio if perf.sharpe_ratio is not None else 0.0
        sharpe_score = min(1.0, max(0.0, (sharpe_val + SHARPE_NORMALIZE_MIN) / (SHARPE_NORMALIZE_MAX - SHARPE_NORMALIZE_MIN)))  # Normalize around 0

        # Risk-adjusted score (lower drawdown is better) with None check
        if perf.max_drawdown is None:
            risk_score = RISK_SCORE_DEFAULT  # Default neutral score if drawdown unknown
        else:
            risk_score = max(0.0, 1.0 - (perf.max_drawdown / MAX_DRAWDOWN_ACCEPTABLE))  # Assume MAX_DRAWDOWN_ACCEPTABLE max acceptable drawdown

        # Weighted combination
        composite_score = (
            win_rate_score * PERF_WIN_RATE_WEIGHT +
            profit_factor_score * PERF_PROFIT_FACTOR_WEIGHT +
            sharpe_score * PERF_SHARPE_WEIGHT +
            risk_score * PERF_RISK_WEIGHT
        )

        return composite_score

    def calculate_strategy_weight(
        self,
        strategy_meta: StrategyMetadata,
        knowledge_entries: List[KnowledgeEntry],
        current_market: MarketCondition,
        market_similarity_cache: Dict[MarketCondition, float]
    ) -> float:
        """
        Calculate adaptive weight for a strategy based on its knowledge.

        Args:
            strategy_meta: Strategy metadata
            knowledge_entries: List of relevant knowledge entries
            current_market: Current market conditions
            market_similarity_cache: Precomputed market similarities

        Returns:
            Adaptive weight for the strategy
        """
        if not knowledge_entries:
            return NO_KNOWLEDGE_DEFAULT_WEIGHT

        # Aggregate performance metrics across all relevant entries
        total_weighted_score = 0.0
        total_weight = 0.0

        for entry in knowledge_entries:
            # Retrieve precomputed market similarity from cache
            market_similarity = market_similarity_cache[entry.market_condition]

            # Calculate recency weight
            recency_weight = self.calculate_recency_weight(entry.last_updated)

            # Calculate sample size weight
            sample_weight = self.calculate_sample_size_weight(entry.sample_size)

            # Combine weights
            combined_weight = (
                market_similarity * self.regime_similarity_weight +
                recency_weight * self.recency_weight +
                sample_weight * self.sample_size_weight
            )

            # Performance score based on multiple metrics
            performance_score = self.calculate_performance_score(entry.performance)

            # Apply combined weight to performance score
            weighted_score = performance_score * combined_weight
            total_weighted_score += weighted_score
            total_weight += combined_weight

        if total_weight == 0:
            return NO_KNOWLEDGE_DEFAULT_WEIGHT

        # Average weighted score
        avg_score = total_weighted_score / total_weight

        # Convert score to weight multiplier (NO_KNOWLEDGE_DEFAULT_WEIGHT = neutral, >NO_KNOWLEDGE_DEFAULT_WEIGHT = boost, <NO_KNOWLEDGE_DEFAULT_WEIGHT = reduce)
        weight_multiplier = NO_KNOWLEDGE_DEFAULT_WEIGHT + (avg_score - SCORE_NEUTRAL) * WEIGHT_MULTIPLIER_SCALE  # Scale factor

        return self._clamp_weight(weight_multiplier)

    def _clamp_weight(self, weight: float) -> float:
        """
        Clamp weight to configured min/max values.

        Args:
            weight: Weight value to clamp

        Returns:
            Clamped weight value
        """
        return max(self.min_weight, min(self.max_weight, weight))

    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize weights to maintain relative relationships using sum-based normalization.

        Args:
            weights: Dictionary of strategy weights

        Returns:
            Normalized weights dictionary
        """
        if not weights:
            return weights

        # Calculate sum of all weights
        total = sum(weights.values())

        if total == 0:
            # Handle edge case where all weights are zero - distribute equally
            num_weights = len(weights)
            normalized = {k: 1.0 / num_weights for k in weights}
        else:
            # Standard sum-based normalization: divide each weight by total sum
            normalized = {k: v / total for k, v in weights.items()}

        # Ensure the sum of normalized weights equals 1.0 (within floating-point tolerance)
        total_normalized = sum(normalized.values())
        assert abs(total_normalized - 1.0) < NORMALIZATION_TOLERANCE, f"Normalization failed: sum = {total_normalized}, expected 1.0"

        return normalized


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation using OrderedDict.

    This cache automatically evicts the least recently used items when the
    maximum size is exceeded, preventing memory exhaustion in long-running
    applications with large numbers of cached items.

    For production systems with multiple application instances or very large
    datasets, consider using a distributed cache like Redis, which provides:
    - Shared caching across multiple application instances
    - Persistence to disk for cache survival across restarts
    - Advanced features like pub/sub for cache invalidation
    - Better memory management for very large cache sizes
    - Built-in clustering and high availability
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Dict[str, float]] = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, float]]:
        """
        Retrieve an item from the cache.

        Args:
            key: Cache key to look up

        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Dict[str, float]) -> None:
        """
        Store an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing item and move to end
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
        self.cache[key] = value

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()

    def size(self) -> int:
        """Get the current number of items in the cache."""
        return len(self.cache)


class CacheManager:
    """
    Manages caching for adaptive weight calculations using LRU eviction.

    This class handles cache storage, retrieval, and invalidation to improve
    performance by avoiding redundant calculations. It uses an LRU cache to
    prevent memory exhaustion and ensure efficient memory usage.
    """

    def __init__(self, ttl_minutes: int = CACHE_TTL_MINUTES_DEFAULT, max_cache_size: int = 1000):
        """
        Initialize the cache manager.

        Args:
            ttl_minutes: Time-to-live for cache entries in minutes
            max_cache_size: Maximum number of cache entries to maintain
        """
        self._weight_cache = LRUCache(max_size=max_cache_size)
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(minutes=ttl_minutes)

    def get_cache_key(
        self,
        market_condition: MarketCondition,
        strategies: List[StrategyMetadata]
    ) -> str:
        """
        Generate cache key for weight calculations.

        Args:
            market_condition: Current market conditions
            strategies: List of strategy metadata

        Returns:
            Cache key string
        """
        # Use structured key to prevent collisions: concatenate components with separators
        # This ensures uniqueness and readability, avoiding potential hash-based collisions
        # by creating a deterministic, human-readable key that uniquely identifies the input combination
        strategy_names = sorted([s.name for s in strategies])
        regime = market_condition.regime.value
        return f"{regime}:{'_'.join(strategy_names)}:{market_condition.timestamp.date()}"

    def is_cache_valid(self) -> bool:
        """
        Check if cache is still valid.

        Returns:
            True if cache is valid, False otherwise
        """
        if self._cache_timestamp is None:
            return False
        return (datetime.now() - self._cache_timestamp) < self._cache_ttl

    def get_cached_weights(self, cache_key: str) -> Optional[Dict[str, float]]:
        """
        Retrieve cached weights for a given key.

        Args:
            cache_key: Cache key to look up

        Returns:
            Cached weights dictionary or None if not found
        """
        if self.is_cache_valid():
            return self._weight_cache.get(cache_key)
        return None

    def cache_weights(self, cache_key: str, weights: Dict[str, float]) -> None:
        """
        Store weights in cache.

        Args:
            cache_key: Cache key
            weights: Weights to cache
        """
        self._weight_cache.put(cache_key, weights)
        self._cache_timestamp = datetime.now()

    def clear_cache(self) -> None:
        """Clear all cached weights."""
        self._weight_cache.clear()
        self._cache_timestamp = None

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': self._weight_cache.size(),
            'cache_age_minutes': (
                (datetime.now() - self._cache_timestamp).total_seconds() / 60
                if self._cache_timestamp else None
            ),
            'cache_ttl_minutes': self._cache_ttl.total_seconds() / 60
        }


class AdaptiveWeightingEngine:
    """
    Engine for calculating adaptive weights based on historical knowledge.

    This class orchestrates the adaptive weighting process by coordinating
    between the WeightingCalculator for mathematical computations and
    CacheManager for performance optimization. It implements various weighting
    algorithms that consider historical performance metrics, market regime
    similarity, strategy category effectiveness, recency, and sample size factors.
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

        # Initialize specialized components
        self.calculator = WeightingCalculator(self.config)
        self.cache_manager = CacheManager(ttl_minutes=CACHE_TTL_MINUTES_DEFAULT)

        # Weighting algorithm parameters (for backward compatibility)
        self.performance_weight = self.config.get('performance_weight', 0.4)
        self.regime_similarity_weight = self.config.get('regime_similarity_weight', 0.3)
        self.recency_weight = self.config.get('recency_weight', 0.2)
        self.sample_size_weight = self.config.get('sample_size_weight', 0.1)

        # Decay factors
        self.performance_decay_days = self.config.get('performance_decay_days', 90)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)

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

        This method has been refactored to use parallel processing for improved performance
        under high load with many strategies. The weight calculations for each strategy
        are independent and can be executed concurrently using ThreadPoolExecutor.

        Benefits of parallel processing:
        - Reduces total computation time when processing many strategies
        - Keeps the main thread responsive by delegating CPU-bound tasks
        - Scales better with increasing number of strategies
        - Maintains thread safety since each strategy calculation is independent

        Args:
            current_market: Current market conditions
            available_strategies: List of available strategy metadata
            base_weights: Optional base weights to adjust

        Returns:
            Dictionary mapping strategy names to adaptive weights
        """
        logger.info(f"Starting adaptive weight calculation for {len(available_strategies)} strategies in {current_market.regime.value} regime")

        # Check cache first with error handling and recovery
        cache_key = self.cache_manager.get_cache_key(current_market, available_strategies)
        cached_weights = None
        try:
            cached_weights = self.cache_manager.get_cached_weights(cache_key)
            if cached_weights:
                logger.info(f"Retrieved cached weights for cache key: {cache_key}")
                return cached_weights
        except (KeyError, ValueError) as e:
            logger.warning(f"Cache lookup failed for key {cache_key}: {e}. Falling back to database query.")
        except Exception as e:
            logger.error(f"Unexpected error during cache lookup for key {cache_key}: {e}. Falling back to database query.")

        # Initialize weights
        adaptive_weights = base_weights.copy() if base_weights else {}
        strategy_names = [s.name for s in available_strategies]

        for strategy_name in strategy_names:
            if strategy_name not in adaptive_weights:
                adaptive_weights[strategy_name] = NO_KNOWLEDGE_DEFAULT_WEIGHT

        # Query relevant knowledge entries
        relevant_knowledge = self._query_relevant_knowledge(current_market, available_strategies)

        if not relevant_knowledge.entries:
            logger.info("No relevant knowledge found, using base weights")
            return adaptive_weights

        # Precompute market similarities for performance optimization
        market_similarity_cache = self._build_market_similarity_cache(
            current_market, relevant_knowledge.entries
        )

        # Parallel weight calculation using ThreadPoolExecutor
        # This improves performance by processing multiple strategies concurrently
        # Each strategy's weight calculation is independent, making it ideal for parallelization
        with ThreadPoolExecutor(max_workers=min(len(available_strategies), 10)) as executor:
            # Submit all weight calculation tasks
            future_to_strategy = {
                executor.submit(
                    self._calculate_single_strategy_weight,
                    strategy_meta,
                    relevant_knowledge,
                    current_market,
                    market_similarity_cache
                ): strategy_meta for strategy_meta in available_strategies
            }

            # Collect results as they complete
            for future in as_completed(future_to_strategy):
                strategy_meta = future_to_strategy[future]
                try:
                    strategy_name, weight = future.result()
                    adaptive_weights[strategy_name] = weight
                except Exception as exc:
                    logger.error(f"Strategy {strategy_meta.name} weight calculation failed: {exc}")
                    # Fallback to default weight
                    adaptive_weights[strategy_meta.name] = NO_KNOWLEDGE_DEFAULT_WEIGHT

        # Normalize weights to maintain relative relationships
        adaptive_weights = self.calculator.normalize_weights(adaptive_weights)

        # Cache the results with error handling
        try:
            self.cache_manager.cache_weights(cache_key, adaptive_weights)
            logger.info(f"Cached adaptive weights for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to cache weights for key {cache_key}: {e}. Continuing without caching.")

        logger.info(f"Successfully calculated adaptive weights for {len(available_strategies)} strategies")
        return adaptive_weights

    def _calculate_single_strategy_weight(
        self,
        strategy_meta: StrategyMetadata,
        relevant_knowledge: KnowledgeQueryResult,
        current_market: MarketCondition,
        market_similarity_cache: Dict[MarketCondition, float]
    ) -> Tuple[str, float]:
        """
        Calculate weight for a single strategy (used for parallel processing).

        Args:
            strategy_meta: Strategy metadata
            relevant_knowledge: Relevant knowledge entries
            current_market: Current market conditions
            market_similarity_cache: Precomputed market similarities

        Returns:
            Tuple of (strategy_name, weight)
        """
        strategy_name = strategy_meta.name
        strategy_knowledge = self._filter_strategy_knowledge(relevant_knowledge, strategy_name)

        if strategy_knowledge:
            adaptive_weight = self.calculator.calculate_strategy_weight(
                strategy_meta, strategy_knowledge, current_market, market_similarity_cache
            )
        else:
            # Reduce weight for strategies with no historical knowledge
            adaptive_weight = NO_KNOWLEDGE_DEFAULT_WEIGHT * NO_STRATEGY_KNOWLEDGE_MULTIPLIER

        return strategy_name, adaptive_weight

    def _build_market_similarity_cache(
        self,
        current_market: MarketCondition,
        knowledge_entries: List[KnowledgeEntry]
    ) -> Dict[MarketCondition, float]:
        """
        Build cache of market similarities for performance optimization.

        Args:
            current_market: Current market conditions
            knowledge_entries: List of knowledge entries

        Returns:
            Dictionary mapping market conditions to similarity scores
        """
        market_similarity_cache: Dict[MarketCondition, float] = {}
        for entry in knowledge_entries:
            historical_market = entry.market_condition
            if historical_market not in market_similarity_cache:
                similarity = self.calculator.calculate_market_similarity(
                    current_market, historical_market
                )
                market_similarity_cache[historical_market] = similarity
        return market_similarity_cache

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
            min_sample_size=self.config.get('min_sample_size', MIN_SAMPLE_SIZE_DEFAULT),
            limit=QUERY_LIMIT  # Get more data for better analysis
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



    def get_strategy_recommendations(
        self,
        current_market: MarketCondition,
        available_strategies: List[StrategyMetadata],
        top_n: int = TOP_N_DEFAULT
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
        if weight > CONFIDENCE_HIGH:
            confidence = "High confidence"
        elif weight > CONFIDENCE_MODERATE:
            confidence = "Moderate confidence"
        elif weight > CONFIDENCE_LOW:
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
        logger.info(f"Starting knowledge update from trade for strategy {strategy_name} in {market_condition.regime.value} regime")
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
                    confidence_score=INITIAL_CONFIDENCE,  # Initial confidence
                    sample_size=1,
                    last_updated=datetime.now()
                )

                success = self.storage.save_entry(new_entry)
                action = "created"

            if success:
                logger.info(f"Knowledge entry {action} for strategy {strategy_name}")
                # Invalidate cache since knowledge has changed
                self.cache_manager.clear_cache()

            return success

        except (ValueError, KeyError) as e:
            logger.error(f"Invalid data or missing key when updating knowledge from trade for strategy {strategy_name}: {e}")
            raise ValueError(f"Failed to update knowledge due to invalid data: {e}") from e
        except IOError as e:
            logger.error(f"Storage I/O error when updating knowledge from trade for strategy {strategy_name}: {e}")
            raise IOError(f"Failed to update knowledge due to storage error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error when updating knowledge from trade for strategy {strategy_name}: {e}")
            raise RuntimeError(f"Unexpected error during knowledge update: {e}") from e

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
            profit_factor=max(PROFIT_FACTOR_MIN, abs(trade_result.get('pnl', 0)) / max(ENTRY_PRICE_MIN, abs(trade_result.get('entry_price', 1)))),
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

        # Add cache statistics from CacheManager
        cache_stats = self.cache_manager.get_cache_stats()
        stats.update(cache_stats)

        # Add adaptive-specific stats
        stats.update({
            'performance_weight': self.performance_weight,
            'regime_similarity_weight': self.regime_similarity_weight,
            'recency_weight': self.recency_weight,
            'sample_size_weight': self.sample_size_weight
        })

        return stats
