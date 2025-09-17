"""
Knowledge Base Schema Definitions

This module defines the standardized data structures for storing and retrieving
trading knowledge entries. It provides schemas for market regimes, strategy
metadata, performance metrics, and outcome classifications.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING = "trending"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE_TIGHT = "range_tight"
    RANGE_WIDE = "range_wide"
    VOLATILE_SPIKE = "volatile_spike"
    UNKNOWN = "unknown"


class StrategyCategory(Enum):
    """Strategy category classifications."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME_BASED = "volume_based"


class OutcomeTag(Enum):
    """Outcome classification tags."""
    SUCCESS = "success"
    FAILURE = "failure"
    BREAK_EVEN = "break_even"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIMEOUT = "timeout"
    LIQUIDATION = "liquidation"
    MANUAL_CLOSE = "manual_close"


class ValidationError(Exception):
    """Custom exception for knowledge entry validation failures."""
    pass


@dataclass
class MarketCondition:
    """Represents market conditions at a point in time."""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    liquidity_score: Optional[float] = None
    volume_trend: Optional[str] = None
    price_range: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'regime': self.regime.value,
            'volatility': self.volatility,
            'trend_strength': self.trend_strength,
            'liquidity_score': self.liquidity_score,
            'volume_trend': self.volume_trend,
            'price_range': self.price_range,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MarketCondition:
        """Create from dictionary."""
        return cls(
            regime=MarketRegime(data['regime']),
            volatility=data['volatility'],
            trend_strength=data['trend_strength'],
            liquidity_score=data.get('liquidity_score'),
            volume_trend=data.get('volume_trend'),
            price_range=data.get('price_range'),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class StrategyMetadata:
    """Metadata about a trading strategy."""
    name: str
    category: StrategyCategory
    parameters: Dict[str, Any]
    timeframe: str
    indicators_used: List[str]
    risk_profile: str  # 'low', 'medium', 'high'
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'category': self.category.value,
            'parameters': self.parameters,
            'timeframe': self.timeframe,
            'indicators_used': self.indicators_used,
            'risk_profile': self.risk_profile,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StrategyMetadata:
        """Create from dictionary."""
        return cls(
            name=data['name'],
            category=StrategyCategory(data['category']),
            parameters=data['parameters'],
            timeframe=data['timeframe'],
            indicators_used=data['indicators_used'],
            risk_profile=data['risk_profile'],
            description=data.get('description')
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics for a trading strategy."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    total_pnl: float
    total_returns: float
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    recovery_factor: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'total_pnl': self.total_pnl,
            'total_returns': self.total_returns,
            'calmar_ratio': self.calmar_ratio,
            'sortino_ratio': self.sortino_ratio,
            'recovery_factor': self.recovery_factor
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PerformanceMetrics:
        """Create from dictionary."""
        return cls(
            total_trades=data['total_trades'],
            winning_trades=data['winning_trades'],
            losing_trades=data['losing_trades'],
            win_rate=data['win_rate'],
            profit_factor=data['profit_factor'],
            sharpe_ratio=data['sharpe_ratio'],
            max_drawdown=data['max_drawdown'],
            avg_win=data['avg_win'],
            avg_loss=data['avg_loss'],
            total_pnl=data['total_pnl'],
            total_returns=data['total_returns'],
            calmar_ratio=data.get('calmar_ratio'),
            sortino_ratio=data.get('sortino_ratio'),
            recovery_factor=data.get('recovery_factor')
        )


@dataclass
class KnowledgeEntry:
    """Complete knowledge entry for a strategy-market condition combination."""
    id: str
    market_condition: MarketCondition
    strategy_metadata: StrategyMetadata
    performance: PerformanceMetrics
    outcome: OutcomeTag
    confidence_score: float
    sample_size: int
    last_updated: datetime
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'market_condition': self.market_condition.to_dict(),
            'strategy_metadata': self.strategy_metadata.to_dict(),
            'performance': self.performance.to_dict(),
            'outcome': self.outcome.value,
            'confidence_score': self.confidence_score,
            'sample_size': self.sample_size,
            'last_updated': self.last_updated.isoformat(),
            'tags': self.tags,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeEntry:
        """Create from dictionary."""
        return cls(
            id=data['id'],
            market_condition=MarketCondition.from_dict(data['market_condition']),
            strategy_metadata=StrategyMetadata.from_dict(data['strategy_metadata']),
            performance=PerformanceMetrics.from_dict(data['performance']),
            outcome=OutcomeTag(data['outcome']),
            confidence_score=data['confidence_score'],
            sample_size=data['sample_size'],
            last_updated=datetime.fromisoformat(data['last_updated']),
            tags=data.get('tags', []),
            notes=data.get('notes')
        )

    def update_performance(self, new_performance: PerformanceMetrics):
        """Update performance metrics and recalculate confidence."""
        # Weighted average update
        total_samples = self.sample_size + 1
        weight_old = self.sample_size / total_samples
        weight_new = 1 / total_samples

        # Update metrics using weighted average
        self.performance.total_trades = int(
            self.performance.total_trades * weight_old +
            new_performance.total_trades * weight_new
        )
        self.performance.winning_trades = int(
            self.performance.winning_trades * weight_old +
            new_performance.winning_trades * weight_new
        )
        self.performance.losing_trades = int(
            self.performance.losing_trades * weight_old +
            new_performance.losing_trades * weight_new
        )
        self.performance.win_rate = (
            self.performance.win_rate * weight_old +
            new_performance.win_rate * weight_new
        )
        self.performance.profit_factor = (
            self.performance.profit_factor * weight_old +
            new_performance.profit_factor * weight_new
        )
        self.performance.sharpe_ratio = (
            self.performance.sharpe_ratio * weight_old +
            new_performance.sharpe_ratio * weight_new
        )
        self.performance.max_drawdown = max(
            self.performance.max_drawdown,
            new_performance.max_drawdown
        )
        self.performance.avg_win = (
            self.performance.avg_win * weight_old +
            new_performance.avg_win * weight_new
        )
        self.performance.avg_loss = (
            self.performance.avg_loss * weight_old +
            new_performance.avg_loss * weight_new
        )
        self.performance.total_pnl = (
            self.performance.total_pnl * weight_old +
            new_performance.total_pnl * weight_new
        )
        self.performance.total_returns = (
            self.performance.total_returns * weight_old +
            new_performance.total_returns * weight_new
        )

        self.sample_size = total_samples
        self.last_updated = datetime.now()

        # Recalculate confidence based on sample size and consistency
        self._recalculate_confidence()

    def _recalculate_confidence(self):
        """Recalculate confidence score based on sample size and performance consistency."""
        # Base confidence on sample size (more samples = higher confidence)
        sample_confidence = min(1.0, self.sample_size / 50)  # 50 samples for max confidence

        # Performance consistency factor
        if self.performance.total_trades > 5:
            # High win rate and profit factor indicate consistent performance
            performance_consistency = min(1.0, (
                self.performance.win_rate * 0.6 +
                min(1.0, self.performance.profit_factor / 2) * 0.4
            ))
        else:
            performance_consistency = 0.5  # Neutral for small sample sizes

        self.confidence_score = (sample_confidence * 0.7 + performance_consistency * 0.3)


@dataclass
class KnowledgeQuery:
    """Query structure for retrieving knowledge entries."""
    market_regime: Optional[MarketRegime] = None
    strategy_name: Optional[str] = None
    strategy_category: Optional[StrategyCategory] = None
    min_confidence: float = 0.0
    min_sample_size: int = 1
    timeframe: Optional[str] = None
    tags: Optional[List[str]] = None
    limit: int = 10
    sort_by: str = "confidence_score"  # confidence_score, win_rate, profit_factor, sample_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'market_regime': self.market_regime.value if self.market_regime else None,
            'strategy_name': self.strategy_name,
            'strategy_category': self.strategy_category.value if self.strategy_category else None,
            'min_confidence': self.min_confidence,
            'min_sample_size': self.min_sample_size,
            'timeframe': self.timeframe,
            'tags': self.tags,
            'limit': self.limit,
            'sort_by': self.sort_by
        }


@dataclass
class KnowledgeQueryResult:
    """Result of a knowledge query."""
    entries: List[KnowledgeEntry]
    total_found: int
    query: KnowledgeQuery
    execution_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'entries': [entry.to_dict() for entry in self.entries],
            'total_found': self.total_found,
            'query': self.query.to_dict(),
            'execution_time': self.execution_time
        }


def validate_knowledge_entry(entry: KnowledgeEntry) -> List[str]:
    """
    Validate a knowledge entry and return a list of validation errors.

    This function performs comprehensive validation of knowledge entry data to ensure
    data integrity and prevent corrupted entries from being stored. Validation failures
    are logged with detailed error information for debugging and audit trails.

    Args:
        entry: The KnowledgeEntry to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    if not entry.id:
        errors.append("Entry ID cannot be empty")

    if entry.confidence_score < 0 or entry.confidence_score > 1:
        errors.append(f"Confidence score {entry.confidence_score} must be between 0 and 1")

    if entry.sample_size < 1:
        errors.append(f"Sample size {entry.sample_size} must be at least 1")

    if entry.performance.total_trades < 0:
        errors.append(f"Total trades {entry.performance.total_trades} cannot be negative")

    if entry.performance.win_rate < 0 or entry.performance.win_rate > 1:
        errors.append(f"Win rate {entry.performance.win_rate} must be between 0 and 1")

    if errors:
        error_message = f"Validation failed for entry '{entry.id}': {'; '.join(errors)}"
        logging.error(error_message)

    return errors
