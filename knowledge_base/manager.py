"""
Knowledge Base Manager

This module provides the central interface for managing the trading knowledge base.
It orchestrates the full knowledge flow including storage, retrieval, and adaptive
weighting for strategy selection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
import threading

from .schema import (
    KnowledgeEntry, KnowledgeQuery, KnowledgeQueryResult,
    MarketRegime, StrategyCategory, MarketCondition, StrategyMetadata,
    PerformanceMetrics, OutcomeTag, validate_knowledge_entry
)
from .storage import KnowledgeStorage
from .adaptive import AdaptiveWeightingEngine

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    Central manager for the trading knowledge base system.

    This class provides a unified interface for:
    - Storing and retrieving trading knowledge
    - Adaptive strategy weighting
    - Performance tracking and analytics
    - Knowledge base maintenance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge manager.

        Args:
            config: Configuration dictionary with knowledge base settings
        """
        self.config = config or self._get_default_config()
        self.enabled = self.config.get('enabled', True)

        if not self.enabled:
            logger.info("Knowledge base is disabled")
            self.storage = None
            self.adaptive_engine = None
            return

        # Initialize storage backend
        storage_config = self.config.get('storage', {})
        backend = storage_config.pop('backend', 'json')  # Remove backend from config to avoid conflict
        self.storage = KnowledgeStorage(backend, **storage_config)

        # Initialize adaptive weighting engine
        adaptive_config = self.config.get('adaptive', {})
        self.adaptive_engine = AdaptiveWeightingEngine(self.storage, adaptive_config)

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"KnowledgeManager initialized with {backend} storage backend")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'storage': {
                'backend': 'json',
                'file_path': 'knowledge_base/knowledge.json',
                'compress': False
            },
            'adaptive': {
                'performance_weight': 0.4,
                'regime_similarity_weight': 0.3,
                'recency_weight': 0.2,
                'sample_size_weight': 0.1,
                'performance_decay_days': 90,
                'confidence_threshold': 0.3,
                'min_sample_size': 5,
                'max_weight': 3.0,
                'min_weight': 0.1
            },
            'maintenance': {
                'auto_cleanup': True,
                'max_age_days': 365,
                'min_confidence_cleanup': 0.1
            }
        }

    def store_trade_knowledge(
        self,
        strategy_name: str,
        market_condition: MarketCondition,
        trade_result: Dict[str, Any],
        strategy_metadata: Optional[StrategyMetadata] = None
    ) -> bool:
        """
        Store knowledge from a completed trade.

        Args:
            strategy_name: Name of the strategy used
            market_condition: Market conditions during the trade
            trade_result: Trade performance data
            strategy_metadata: Optional strategy metadata

        Returns:
            Success status
        """
        if not self.enabled or not self.adaptive_engine:
            return False

        with self._lock:
            try:
                # Use adaptive engine to update knowledge
                success = self.adaptive_engine.update_knowledge_from_trade(
                    strategy_name, market_condition, trade_result
                )

                if success:
                    logger.info(f"Stored knowledge for strategy {strategy_name}")
                else:
                    logger.warning(f"Failed to store knowledge for strategy {strategy_name}")

                return success

            except Exception as e:
                logger.error(f"Error storing trade knowledge: {e}")
                return False

    def get_adaptive_weights(
        self,
        current_market: MarketCondition,
        available_strategies: List[StrategyMetadata],
        base_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get adaptive weights for strategies based on historical knowledge.

        Args:
            current_market: Current market conditions
            available_strategies: List of available strategy metadata
            base_weights: Optional base weights to adjust

        Returns:
            Dictionary mapping strategy names to adaptive weights
        """
        if not self.enabled or not self.adaptive_engine:
            # Return equal weights if knowledge base is disabled
            return {s.name: 1.0 for s in available_strategies}

        with self._lock:
            try:
                return self.adaptive_engine.calculate_adaptive_weights(
                    current_market, available_strategies, base_weights
                )
            except Exception as e:
                logger.error(f"Error calculating adaptive weights: {e}")
                return {s.name: 1.0 for s in available_strategies}

    def get_strategy_recommendations(
        self,
        current_market: MarketCondition,
        available_strategies: List[StrategyMetadata],
        top_n: int = 3
    ) -> List[Tuple[str, float, str]]:
        """
        Get top strategy recommendations with reasoning.

        Args:
            current_market: Current market conditions
            available_strategies: Available strategies
            top_n: Number of recommendations to return

        Returns:
            List of (strategy_name, weight, reasoning) tuples
        """
        if not self.enabled or not self.adaptive_engine:
            # Return basic recommendations
            recommendations = []
            for i, strategy in enumerate(available_strategies[:top_n]):
                recommendations.append((
                    strategy.name,
                    1.0,
                    f"Basic recommendation for {strategy.name} (knowledge base disabled)"
                ))
            return recommendations

        with self._lock:
            try:
                return self.adaptive_engine.get_strategy_recommendations(
                    current_market, available_strategies, top_n
                )
            except Exception as e:
                logger.error(f"Error getting strategy recommendations: {e}")
                return []

    def query_knowledge(
        self,
        query: KnowledgeQuery
    ) -> KnowledgeQueryResult:
        """
        Query the knowledge base for specific entries.

        Args:
            query: Knowledge query parameters

        Returns:
            Query results
        """
        if not self.enabled or not self.storage:
            return KnowledgeQueryResult([], 0, query, 0.0)

        with self._lock:
            try:
                return self.storage.query_entries(query)
            except Exception as e:
                logger.error(f"Error querying knowledge: {e}")
                return KnowledgeQueryResult([], 0, query, 0.0)

    def get_knowledge_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Retrieve a specific knowledge entry by ID.

        Args:
            entry_id: Knowledge entry ID

        Returns:
            Knowledge entry or None
        """
        if not self.enabled or not self.storage:
            return None

        with self._lock:
            try:
                return self.storage.get_entry(entry_id)
            except Exception as e:
                logger.error(f"Error retrieving knowledge entry {entry_id}: {e}")
                return None

    def update_knowledge_entry(
        self,
        entry_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update an existing knowledge entry.

        Args:
            entry_id: Knowledge entry ID
            updates: Fields to update

        Returns:
            Success status
        """
        if not self.enabled or not self.storage:
            return False

        with self._lock:
            try:
                entry = self.storage.get_entry(entry_id)
                if not entry:
                    return False

                # Apply updates
                for key, value in updates.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)

                # Validate updated entry
                errors = validate_knowledge_entry(entry)
                if errors:
                    logger.error(f"Invalid updates for entry {entry_id}: {errors}")
                    return False

                # Save updated entry
                success = self.storage.save_entry(entry)

                if success:
                    # Invalidate adaptive engine cache
                    if self.adaptive_engine:
                        self.adaptive_engine._weight_cache.clear()
                        self.adaptive_engine._cache_timestamp = None

                return success

            except Exception as e:
                logger.error(f"Error updating knowledge entry {entry_id}: {e}")
                return False

    def delete_knowledge_entry(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry.

        Args:
            entry_id: Knowledge entry ID

        Returns:
            Success status
        """
        if not self.enabled or not self.storage:
            return False

        with self._lock:
            try:
                success = self.storage.delete_entry(entry_id)

                if success and self.adaptive_engine:
                    # Invalidate cache
                    self.adaptive_engine._weight_cache.clear()
                    self.adaptive_engine._cache_timestamp = None

                return success

            except Exception as e:
                logger.error(f"Error deleting knowledge entry {entry_id}: {e}")
                return False

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge base.

        Returns:
            Statistics dictionary
        """
        if not self.enabled:
            return {'enabled': False}

        stats = {
            'enabled': True,
            'storage_stats': {},
            'adaptive_stats': {},
            'knowledge_summary': {}
        }

        with self._lock:
            try:
                if self.storage:
                    stats['storage_stats'] = self.storage.get_stats()

                if self.adaptive_engine:
                    stats['adaptive_stats'] = self.adaptive_engine.get_adaptive_statistics()

                # Get knowledge summary
                if self.storage:
                    all_entries = self.storage.list_entries(limit=1000)
                    stats['knowledge_summary'] = self._analyze_knowledge_entries(all_entries)

                return stats

            except Exception as e:
                logger.error(f"Error getting knowledge statistics: {e}")
                return {'enabled': True, 'error': str(e)}

    def _analyze_knowledge_entries(self, entries: List[KnowledgeEntry]) -> Dict[str, Any]:
        """Analyze knowledge entries for summary statistics."""
        if not entries:
            return {'total_entries': 0}

        # Group by various dimensions
        by_regime = {}
        by_strategy = {}
        by_outcome = {}

        total_confidence = 0
        total_sample_size = 0

        for entry in entries:
            regime = entry.market_condition.regime.value
            strategy = entry.strategy_metadata.name
            outcome = entry.outcome.value

            # Count by regime
            by_regime[regime] = by_regime.get(regime, 0) + 1

            # Count by strategy
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1

            # Count by outcome
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

            total_confidence += entry.confidence_score
            total_sample_size += entry.sample_size

        return {
            'total_entries': len(entries),
            'avg_confidence': total_confidence / len(entries),
            'avg_sample_size': total_sample_size / len(entries),
            'regime_distribution': by_regime,
            'strategy_distribution': by_strategy,
            'outcome_distribution': by_outcome,
            'unique_strategies': len(by_strategy),
            'unique_regimes': len(by_regime)
        }

    def perform_maintenance(self) -> Dict[str, Any]:
        """
        Perform maintenance operations on the knowledge base.

        Returns:
            Maintenance results
        """
        if not self.enabled:
            return {'enabled': False}

        results = {
            'cleanup_performed': False,
            'entries_removed': 0,
            'errors': []
        }

        with self._lock:
            try:
                maintenance_config = self.config.get('maintenance', {})

                if maintenance_config.get('auto_cleanup', True):
                    # Remove old entries
                    max_age_days = maintenance_config.get('max_age_days', 365)
                    min_confidence = maintenance_config.get('min_confidence_cleanup', 0.1)

                    cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=max_age_days)

                    all_entries = self.storage.list_entries(limit=10000)
                    removed_count = 0

                    for entry in all_entries:
                        should_remove = (
                            entry.last_updated < cutoff_date and
                            entry.confidence_score < min_confidence
                        )

                        if should_remove:
                            if self.storage.delete_entry(entry.id):
                                removed_count += 1
                            else:
                                results['errors'].append(f"Failed to remove entry {entry.id}")

                    results['cleanup_performed'] = True
                    results['entries_removed'] = removed_count

                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} old/low-confidence knowledge entries")

                # Invalidate caches after maintenance
                if self.adaptive_engine:
                    self.adaptive_engine._weight_cache.clear()
                    self.adaptive_engine._cache_timestamp = None

            except Exception as e:
                logger.error(f"Error during knowledge base maintenance: {e}")
                results['errors'].append(str(e))

        return results

    def export_knowledge(self, file_path: Union[str, Path], format: str = 'json') -> bool:
        """
        Export knowledge base to a file.

        Args:
            file_path: Path to export file
            format: Export format ('json', 'csv')

        Returns:
            Success status
        """
        if not self.enabled or not self.storage:
            return False

        try:
            all_entries = self.storage.list_entries(limit=10000)

            if format.lower() == 'json':
                data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_entries': len(all_entries),
                    'entries': [entry.to_dict() for entry in all_entries]
                }

                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)

            elif format.lower() == 'csv':
                if not all_entries:
                    return False

                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    # Write header
                    header = ['id', 'regime', 'strategy_name', 'win_rate', 'profit_factor',
                             'confidence_score', 'sample_size', 'last_updated']
                    writer.writerow(header)

                    # Write data
                    for entry in all_entries:
                        row = [
                            entry.id,
                            entry.market_condition.regime.value,
                            entry.strategy_metadata.name,
                            entry.performance.win_rate,
                            entry.performance.profit_factor,
                            entry.confidence_score,
                            entry.sample_size,
                            entry.last_updated.isoformat()
                        ]
                        writer.writerow(row)

            else:
                logger.error(f"Unsupported export format: {format}")
                return False

            logger.info(f"Exported {len(all_entries)} knowledge entries to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error exporting knowledge: {e}")
            return False

    def import_knowledge(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Import knowledge from a file.

        Args:
            file_path: Path to import file

        Returns:
            Import results
        """
        if not self.enabled or not self.storage:
            return {'success': False, 'error': 'Knowledge base disabled'}

        results = {
            'success': False,
            'entries_imported': 0,
            'entries_skipped': 0,
            'errors': []
        }

        try:
            file_path = Path(file_path)

            if not file_path.exists():
                results['error'] = f"File not found: {file_path}"
                return results

            if file_path.suffix.lower() == '.json':
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                entries_data = data.get('entries', [])

            elif file_path.suffix.lower() == '.csv':
                import csv
                entries_data = []

                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert CSV row to knowledge entry format
                        # This is a simplified conversion
                        entry_data = {
                            'id': row.get('id', ''),
                            'market_condition': {
                                'regime': row.get('regime', 'unknown'),
                                'volatility': float(row.get('volatility', 0.0)),
                                'trend_strength': float(row.get('trend_strength', 0.0)),
                                'timestamp': row.get('timestamp', datetime.now().isoformat())
                            },
                            'strategy_metadata': {
                                'name': row.get('strategy_name', ''),
                                'category': 'trend_following',
                                'parameters': {},
                                'timeframe': '1h',
                                'indicators_used': ['price'],
                                'risk_profile': 'medium'
                            },
                            'performance': {
                                'total_trades': int(row.get('total_trades', 1)),
                                'winning_trades': int(row.get('winning_trades', 0)),
                                'losing_trades': int(row.get('losing_trades', 0)),
                                'win_rate': float(row.get('win_rate', 0.0)),
                                'profit_factor': float(row.get('profit_factor', 1.0)),
                                'sharpe_ratio': float(row.get('sharpe_ratio', 0.0)),
                                'max_drawdown': float(row.get('max_drawdown', 0.0)),
                                'avg_win': float(row.get('avg_win', 0.0)),
                                'avg_loss': float(row.get('avg_loss', 0.0)),
                                'total_pnl': float(row.get('total_pnl', 0.0)),
                                'total_returns': float(row.get('total_returns', 0.0))
                            },
                            'outcome': row.get('outcome', 'success'),
                            'confidence_score': float(row.get('confidence_score', 0.5)),
                            'sample_size': int(row.get('sample_size', 1)),
                            'last_updated': row.get('last_updated', datetime.now().isoformat())
                        }
                        entries_data.append(entry_data)

            else:
                results['error'] = f"Unsupported file format: {file_path.suffix}"
                return results

            # Import entries
            for entry_data in entries_data:
                try:
                    entry = KnowledgeEntry.from_dict(entry_data)

                    # Validate entry
                    errors = validate_knowledge_entry(entry)
                    if errors:
                        results['entries_skipped'] += 1
                        results['errors'].append(f"Invalid entry {entry.id}: {errors}")
                        continue

                    # Save entry
                    if self.storage.save_entry(entry):
                        results['entries_imported'] += 1
                    else:
                        results['entries_skipped'] += 1
                        results['errors'].append(f"Failed to save entry {entry.id}")

                except Exception as e:
                    results['entries_skipped'] += 1
                    results['errors'].append(f"Error processing entry: {e}")

            results['success'] = True

            # Invalidate caches
            if self.adaptive_engine:
                self.adaptive_engine._weight_cache.clear()
                self.adaptive_engine._cache_timestamp = None

            logger.info(f"Imported {results['entries_imported']} knowledge entries from {file_path}")

        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Error importing knowledge: {e}")

        return results

    def reset_knowledge_base(self) -> bool:
        """
        Reset the entire knowledge base (removes all entries).

        Returns:
            Success status
        """
        if not self.enabled or not self.storage:
            return False

        try:
            # Get all entries and delete them
            all_entries = self.storage.list_entries(limit=10000)
            deleted_count = 0

            for entry in all_entries:
                if self.storage.delete_entry(entry.id):
                    deleted_count += 1

            # Clear adaptive engine cache
            if self.adaptive_engine:
                self.adaptive_engine._weight_cache.clear()
                self.adaptive_engine._cache_timestamp = None

            logger.info(f"Reset knowledge base: removed {deleted_count} entries")
            return True

        except Exception as e:
            logger.error(f"Error resetting knowledge base: {e}")
            return False


# Global knowledge manager instance
_knowledge_manager: Optional[KnowledgeManager] = None


def get_knowledge_manager(config: Optional[Dict[str, Any]] = None) -> KnowledgeManager:
    """Get the global knowledge manager instance."""
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = KnowledgeManager(config)
    return _knowledge_manager


def store_trade_result(
    strategy_name: str,
    market_condition: MarketCondition,
    trade_result: Dict[str, Any]
) -> bool:
    """
    Convenience function to store trade knowledge.

    Args:
        strategy_name: Name of the strategy used
        market_condition: Market conditions during the trade
        trade_result: Trade performance data

    Returns:
        Success status
    """
    manager = get_knowledge_manager()
    return manager.store_trade_knowledge(strategy_name, market_condition, trade_result)


def get_adaptive_strategy_weights(
    current_market: MarketCondition,
    available_strategies: List[StrategyMetadata]
) -> Dict[str, float]:
    """
    Convenience function to get adaptive strategy weights.

    Args:
        current_market: Current market conditions
        available_strategies: Available strategies

    Returns:
        Strategy weights dictionary
    """
    manager = get_knowledge_manager()
    return manager.get_adaptive_weights(current_market, available_strategies)
