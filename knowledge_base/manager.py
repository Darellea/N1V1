"""
Knowledge Base Manager

This module provides the central interface for managing the trading knowledge base.
It orchestrates the full knowledge flow including storage, retrieval, and adaptive
weighting for strategy selection.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
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


class KnowledgeValidator:
    """
    Handles input validation and schema validation for knowledge base operations.

    This class centralizes all validation logic to ensure data integrity and
    prevent injection attacks or malformed data from corrupting the knowledge base.
    """

    def __init__(self):
        """Initialize the knowledge validator."""
        self.allowed_update_fields = {
            'market_condition': dict,
            'strategy_metadata': dict,
            'performance': dict,
            'outcome': str,
            'confidence_score': (int, float),
            'sample_size': int,
            'last_updated': str,
            'tags': list,
            'notes': (str, type(None))
        }

    def validate_update_payload(self, updates: Dict[str, Any]) -> List[str]:
        """
        Validate the update payload against the expected schema.

        This function implements robust validation to prevent data corruption and injection attacks
        by ensuring that only expected fields with correct data types are accepted in updates.

        Args:
            updates: Dictionary containing the fields to update

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check for unknown fields
        for key in updates.keys():
            if key not in self.allowed_update_fields:
                errors.append(f"Unknown field '{key}' not allowed in updates")

        # Validate field types
        for key, value in updates.items():
            if key in self.allowed_update_fields:
                expected_type = self.allowed_update_fields[key]
                if not isinstance(value, expected_type):
                    if isinstance(expected_type, tuple):
                        type_names = [t.__name__ for t in expected_type]
                        errors.append(f"Field '{key}' must be one of: {', '.join(type_names)}")
                    else:
                        errors.append(f"Field '{key}' must be of type {expected_type.__name__}")

                # Additional validation for specific fields
                if key == 'confidence_score' and isinstance(value, (int, float)):
                    if not (0.0 <= value <= 1.0):
                        errors.append("Field 'confidence_score' must be between 0.0 and 1.0")

                if key == 'sample_size' and isinstance(value, int):
                    if value < 1:
                        errors.append("Field 'sample_size' must be a positive integer")

                if key == 'outcome' and isinstance(value, str):
                    # Validate against known outcome values if available
                    # For now, just ensure it's a non-empty string
                    if not value.strip():
                        errors.append("Field 'outcome' cannot be empty")

                if key == 'tags' and isinstance(value, list):
                    # Ensure all tags are strings
                    if not all(isinstance(tag, str) for tag in value):
                        errors.append("All tags must be strings")

        return errors

    def validate_knowledge_entry(self, entry: KnowledgeEntry) -> List[str]:
        """
        Validate a complete knowledge entry.

        Args:
            entry: Knowledge entry to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Basic field presence validation
        if not entry.id:
            errors.append("Entry ID is required")

        if not entry.strategy_metadata or not entry.strategy_metadata.name:
            errors.append("Strategy metadata with name is required")

        if not entry.market_condition:
            errors.append("Market condition is required")

        # Validate confidence score range
        if not (0.0 <= entry.confidence_score <= 1.0):
            errors.append("Confidence score must be between 0.0 and 1.0")

        # Validate sample size
        if entry.sample_size < 1:
            errors.append("Sample size must be a positive integer")

        return errors

    def validate_query_parameters(self, query: KnowledgeQuery) -> List[str]:
        """
        Validate query parameters.

        Args:
            query: Knowledge query to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate confidence thresholds
        if query.min_confidence is not None and not (0.0 <= query.min_confidence <= 1.0):
            errors.append("Minimum confidence must be between 0.0 and 1.0")

        if query.max_confidence is not None and not (0.0 <= query.max_confidence <= 1.0):
            errors.append("Maximum confidence must be between 0.0 and 1.0")

        if (query.min_confidence is not None and query.max_confidence is not None and
            query.min_confidence > query.max_confidence):
            errors.append("Minimum confidence cannot be greater than maximum confidence")

        # Validate sample size
        if query.min_sample_size is not None and query.min_sample_size < 1:
            errors.append("Minimum sample size must be a positive integer")

        # Validate limit
        if query.limit is not None and query.limit < 1:
            errors.append("Query limit must be a positive integer")

        return errors


class DataStoreInterface:
    """
    Abstract interface for knowledge base storage operations.

    This class provides a clean abstraction layer over the storage backend,
    allowing for easier testing and potential future storage backend changes.
    """

    def __init__(self, storage: KnowledgeStorage):
        """
        Initialize the data store interface.

        Args:
            storage: The underlying storage backend
        """
        self.storage = storage

    def save_entry(self, entry: KnowledgeEntry) -> bool:
        """
        Save a knowledge entry.

        Args:
            entry: Knowledge entry to save

        Returns:
            Success status
        """
        import asyncio
        return asyncio.run(self.storage.save_entry(entry))

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """
        Retrieve a knowledge entry by ID.

        Args:
            entry_id: Knowledge entry ID

        Returns:
            Knowledge entry or None if not found
        """
        import asyncio
        return asyncio.run(self.storage.get_entry(entry_id))

    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a knowledge entry.

        Args:
            entry_id: Knowledge entry ID

        Returns:
            Success status
        """
        import asyncio
        return asyncio.run(self.storage.delete_entry(entry_id))

    def query_entries(self, query: KnowledgeQuery) -> KnowledgeQueryResult:
        """
        Query knowledge entries.

        Args:
            query: Knowledge query parameters

        Returns:
            Query results
        """
        import asyncio
        return asyncio.run(self.storage.query_entries(query))

    def list_entries(self, limit: Optional[int] = None) -> List[KnowledgeEntry]:
        """
        List knowledge entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of knowledge entries
        """
        import asyncio
        return asyncio.run(self.storage.list_entries(limit))

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Statistics dictionary
        """
        import asyncio
        return asyncio.run(self.storage.get_stats())

    def clear_all(self) -> bool:
        """
        Clear all knowledge entries.

        Returns:
            Success status
        """
        try:
            import asyncio
            all_entries = asyncio.run(self.storage.list_entries(limit=10000))
            deleted_count = 0
            for entry in all_entries:
                if asyncio.run(self.storage.delete_entry(entry.id)):
                    deleted_count += 1
            return deleted_count == len(all_entries)
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False


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
        self.storage = None

        if not self.enabled:
            logger.info("Knowledge base is disabled")
            self.data_store = None
            self.validator = None
            self.adaptive_engine = None
            return

        # Initialize storage backend
        storage_config = self.config.get('storage', {})
        backend = storage_config.pop('backend', 'json')  # Remove backend from config to avoid conflict
        self.storage = KnowledgeStorage(backend, **storage_config)

        # Initialize specialized components
        self.data_store = DataStoreInterface(self.storage)
        self.validator = KnowledgeValidator()

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
                import asyncio
                success = asyncio.run(self.adaptive_engine.update_knowledge_from_trade(
                    strategy_name, market_condition, trade_result
                ))

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
                import asyncio
                return asyncio.run(self.adaptive_engine.calculate_adaptive_weights(
                    current_market, available_strategies, base_weights
                ))
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
                import asyncio
                return asyncio.run(self.adaptive_engine.get_strategy_recommendations(
                    current_market, available_strategies, top_n
                ))
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
        if not self.enabled or not self.data_store:
            return KnowledgeQueryResult([], 0, query, 0.0)

        logger.info(f"Starting knowledge query with limit {query.limit}")

        with self._lock:
            try:
                # Validate query parameters
                validation_errors = self.validator.validate_query_parameters(query)
                if validation_errors:
                    logger.error(f"Invalid query parameters: {validation_errors}")
                    return KnowledgeQueryResult([], 0, query, 0.0)

                return self.data_store.query_entries(query)
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
        if not self.enabled or not self.data_store:
            return None

        with self._lock:
            try:
                return self.data_store.get_entry(entry_id)
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

        This method implements robust input validation for update payloads to prevent
        data corruption, logical errors, and potential injection attacks. The validation
        ensures that only expected fields with correct data types are accepted, providing
        security benefits by rejecting malicious or malformed input.

        Args:
            entry_id: Knowledge entry ID
            updates: Fields to update (must conform to expected schema)

        Returns:
            Success status
        """
        if not self.enabled or not self.data_store:
            return False

        logger.info(f"Starting update of knowledge entry {entry_id} with {len(updates)} fields")

        with self._lock:
            try:
                entry = self.data_store.get_entry(entry_id)
                if not entry:
                    return False

                # Validate the update payload using KnowledgeValidator
                validation_errors = self.validator.validate_update_payload(updates)
                if validation_errors:
                    logger.error(f"Invalid update payload for entry {entry_id}: {validation_errors}")
                    raise ValueError(f"Update validation failed: {', '.join(validation_errors)}")

                # Apply validated updates
                for key, value in updates.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)

                # Validate updated entry using KnowledgeValidator
                errors = self.validator.validate_knowledge_entry(entry)
                if errors:
                    logger.error(f"Invalid updates for entry {entry_id}: {errors}")
                    return False

                # Save updated entry
                success = self.data_store.save_entry(entry)

                if success:
                    # Invalidate adaptive engine cache
                    if self.adaptive_engine:
                        self.adaptive_engine.cache_manager.clear_cache()

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
        if not self.enabled or not self.data_store:
            return False

        with self._lock:
            try:
                success = self.data_store.delete_entry(entry_id)

                if success and self.adaptive_engine:
                    # Invalidate cache
                    self.adaptive_engine.cache_manager.clear_cache()

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
                if self.data_store:
                    stats['storage_stats'] = self.data_store.get_stats()

                if self.adaptive_engine:
                    import asyncio
                    stats['adaptive_stats'] = asyncio.run(self.adaptive_engine.get_adaptive_statistics())

                # Get knowledge summary
                if self.data_store:
                    all_entries = self.data_store.list_entries(limit=1000)
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

                    all_entries = self.data_store.list_entries(limit=10000)
                    removed_count = 0

                    for entry in all_entries:
                        should_remove = (
                            entry.last_updated < cutoff_date and
                            entry.confidence_score < min_confidence
                        )

                        if should_remove:
                            if self.data_store.delete_entry(entry.id):
                                removed_count += 1
                            else:
                                results['errors'].append(f"Failed to remove entry {entry.id}")

                    results['cleanup_performed'] = True
                    results['entries_removed'] = removed_count

                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} old/low-confidence knowledge entries")

                # Invalidate caches after maintenance
                if self.adaptive_engine:
                    self.adaptive_engine.cache_manager.clear_cache()

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
        if not self.enabled or not self.data_store:
            return False

        try:
            all_entries = self.data_store.list_entries(limit=10000)

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
        if not self.enabled or not self.data_store:
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
                    errors = self.validator.validate_knowledge_entry(entry)
                    if errors:
                        results['entries_skipped'] += 1
                        results['errors'].append(f"Invalid entry {entry.id}: {errors}")
                        continue

                    # Save entry
                    if self.data_store.save_entry(entry):
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
                self.adaptive_engine.cache_manager.clear_cache()

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
        if not self.enabled or not self.data_store:
            return False

        try:
            # Use DataStoreInterface's clear_all method for efficient bulk deletion
            success = self.data_store.clear_all()

            # Clear adaptive engine cache
            if self.adaptive_engine:
                self.adaptive_engine.cache_manager.clear_cache()

            if success:
                logger.info("Reset knowledge base: all entries removed")
            return success

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
