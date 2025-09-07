"""
Knowledge Base Package

This package provides adaptive memory and learning capabilities for the trading framework.
It maintains a structured knowledge store that learns from trading history to improve
strategy selection and performance over time.
"""

from .schema import (
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeQueryResult,
    MarketRegime,
    StrategyCategory,
    MarketCondition,
    StrategyMetadata,
    PerformanceMetrics,
    OutcomeTag,
    validate_knowledge_entry
)

from .storage import KnowledgeStorage

from .adaptive import AdaptiveWeightingEngine

from .manager import (
    KnowledgeManager,
    get_knowledge_manager,
    store_trade_result,
    get_adaptive_strategy_weights
)

__all__ = [
    # Schema classes
    'KnowledgeEntry',
    'KnowledgeQuery',
    'KnowledgeQueryResult',
    'MarketRegime',
    'StrategyCategory',
    'MarketCondition',
    'StrategyMetadata',
    'PerformanceMetrics',
    'OutcomeTag',
    'validate_knowledge_entry',

    # Storage
    'KnowledgeStorage',

    # Adaptive engine
    'AdaptiveWeightingEngine',

    # Manager
    'KnowledgeManager',
    'get_knowledge_manager',
    'store_trade_result',
    'get_adaptive_strategy_weights'
]

__version__ = "1.0.0"
