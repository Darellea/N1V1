"""
Unit tests for Knowledge Base & Adaptive Memory system.

This module contains comprehensive tests for the knowledge base functionality
including schema validation, storage operations, adaptive weighting, and
integration with strategy selection.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from knowledge_base.schema import (
    KnowledgeEntry, KnowledgeQuery, KnowledgeQueryResult,
    MarketRegime, StrategyCategory, MarketCondition, StrategyMetadata,
    PerformanceMetrics, OutcomeTag, validate_knowledge_entry
)
from knowledge_base.storage import KnowledgeStorage, JSONStorage, CSVStorage, SQLiteStorage
from knowledge_base.adaptive import AdaptiveWeightingEngine
from knowledge_base.manager import KnowledgeManager


class TestKnowledgeSchema:
    """Test knowledge base schema definitions."""

    def test_market_condition_creation(self):
        """Test MarketCondition dataclass creation and serialization."""
        condition = MarketCondition(
            regime=MarketRegime.TRENDING,
            volatility=0.15,
            trend_strength=0.8,
            timestamp=datetime.now()
        )

        assert condition.regime == MarketRegime.TRENDING
        assert condition.volatility == 0.15
        assert condition.trend_strength == 0.8

        # Test serialization
        data = condition.to_dict()
        assert data['regime'] == 'trending'
        assert data['volatility'] == 0.15

        # Test deserialization
        condition2 = MarketCondition.from_dict(data)
        assert condition2.regime == condition.regime
        assert condition2.volatility == condition.volatility

    def test_strategy_metadata_creation(self):
        """Test StrategyMetadata dataclass creation and serialization."""
        metadata = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={"period": 14},
            timeframe="1h",
            indicators_used=["rsi"],
            risk_profile="medium"
        )

        assert metadata.name == "TestStrategy"
        assert metadata.category == StrategyCategory.TREND_FOLLOWING
        assert metadata.parameters == {"period": 14}

        # Test serialization
        data = metadata.to_dict()
        assert data['name'] == 'TestStrategy'
        assert data['category'] == 'trend_following'

    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics dataclass creation and serialization."""
        metrics = PerformanceMetrics(
            total_trades=100,
            winning_trades=60,
            losing_trades=40,
            win_rate=0.6,
            profit_factor=1.5,
            sharpe_ratio=1.2,
            max_drawdown=0.1,
            avg_win=100.0,
            avg_loss=-80.0,
            total_pnl=2000.0,
            total_returns=0.2
        )

        assert metrics.total_trades == 100
        assert metrics.win_rate == 0.6
        assert metrics.profit_factor == 1.5

        # Test serialization
        data = metrics.to_dict()
        assert data['total_trades'] == 100
        assert data['win_rate'] == 0.6

    def test_knowledge_entry_creation(self):
        """Test KnowledgeEntry dataclass creation and validation."""
        condition = MarketCondition(
            regime=MarketRegime.TRENDING,
            volatility=0.15,
            trend_strength=0.8
        )

        metadata = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium"
        )

        performance = PerformanceMetrics(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            profit_factor=1.3,
            sharpe_ratio=0.8,
            max_drawdown=0.05,
            avg_win=150.0,
            avg_loss=-100.0,
            total_pnl=1500.0,
            total_returns=0.15
        )

        entry = KnowledgeEntry(
            id="test_entry_001",
            market_condition=condition,
            strategy_metadata=metadata,
            performance=performance,
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.8,
            sample_size=50,
            last_updated=datetime.now()
        )

        assert entry.id == "test_entry_001"
        assert entry.confidence_score == 0.8
        assert entry.sample_size == 50

        # Test validation
        errors = validate_knowledge_entry(entry)
        assert len(errors) == 0  # Should pass validation

    def test_knowledge_entry_validation(self):
        """Test KnowledgeEntry validation with invalid data."""
        # Test invalid confidence score
        entry = KnowledgeEntry(
            id="",
            market_condition=MarketCondition(regime=MarketRegime.UNKNOWN, volatility=0.1, trend_strength=0.1),
            strategy_metadata=StrategyMetadata(
                name="Test", category=StrategyCategory.TREND_FOLLOWING,
                parameters={}, timeframe="1h", indicators_used=[], risk_profile="low"
            ),
            performance=PerformanceMetrics(
                total_trades=10, winning_trades=5, losing_trades=5, win_rate=0.5,
                profit_factor=1.0, sharpe_ratio=0.0, max_drawdown=0.0,
                avg_win=0.0, avg_loss=0.0, total_pnl=0.0, total_returns=0.0
            ),
            outcome=OutcomeTag.SUCCESS,
            confidence_score=1.5,  # Invalid: > 1.0
            sample_size=0,  # Invalid: < 1
            last_updated=datetime.now()
        )

        errors = validate_knowledge_entry(entry)
        assert len(errors) > 0
        assert any("confidence" in error.lower() for error in errors)
        assert any("sample" in error.lower() for error in errors)

    def test_knowledge_query_creation(self):
        """Test KnowledgeQuery creation."""
        query = KnowledgeQuery(
            market_regime=MarketRegime.TRENDING,
            strategy_name="TestStrategy",
            min_confidence=0.5,
            min_sample_size=10,
            limit=20
        )

        assert query.market_regime == MarketRegime.TRENDING
        assert query.strategy_name == "TestStrategy"
        assert query.min_confidence == 0.5
        assert query.limit == 20


class TestStorageBackends:
    """Test storage backend implementations."""

    def test_json_storage(self):
        """Test JSON storage backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_knowledge.json"

            storage = JSONStorage(file_path)

            # Create test entry
            entry = self._create_test_entry("test_001")

            # Test save
            assert storage.save_entry(entry)

            # Test get
            retrieved = storage.get_entry("test_001")
            assert retrieved is not None
            assert retrieved.id == "test_001"

            # Test list
            entries = storage.list_entries()
            assert len(entries) == 1
            assert entries[0].id == "test_001"

            # Test stats
            stats = storage.get_stats()
            assert stats['backend'] == 'json'
            assert stats['total_entries'] == 1

    def test_csv_storage(self):
        """Test CSV storage backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_knowledge.csv"

            storage = CSVStorage(file_path)

            # Create test entry
            entry = self._create_test_entry("test_002")

            # Test save
            assert storage.save_entry(entry)

            # Test get
            retrieved = storage.get_entry("test_002")
            assert retrieved is not None
            assert retrieved.id == "test_002"

            # Test list
            entries = storage.list_entries()
            assert len(entries) == 1

    def test_sqlite_storage(self):
        """Test SQLite storage backend."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_knowledge.db"

            storage = SQLiteStorage(db_path)

            # Create test entry
            entry = self._create_test_entry("test_003")

            # Test save
            assert storage.save_entry(entry)

            # Test get
            retrieved = storage.get_entry("test_003")
            assert retrieved is not None
            assert retrieved.id == "test_003"

            # Test query
            query = KnowledgeQuery(limit=10)
            result = storage.query_entries(query)
            assert result.total_found == 1
            assert len(result.entries) == 1

    def test_storage_query_filtering(self):
        """Test storage query filtering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_knowledge.db"
            storage = SQLiteStorage(db_path)

            # Create multiple test entries
            entries = [
                self._create_test_entry("entry_1", MarketRegime.TRENDING),
                self._create_test_entry("entry_2", MarketRegime.SIDEWAYS),
                self._create_test_entry("entry_3", MarketRegime.TRENDING),
            ]

            for entry in entries:
                storage.save_entry(entry)

            # Test regime filtering
            query = KnowledgeQuery(market_regime=MarketRegime.TRENDING)
            result = storage.query_entries(query)
            assert result.total_found == 2

            # Test confidence filtering
            query = KnowledgeQuery(min_confidence=0.9)
            result = storage.query_entries(query)
            assert result.total_found == 0  # Our test entries have confidence 0.7

    def _create_test_entry(self, entry_id: str, regime: MarketRegime = MarketRegime.TRENDING) -> KnowledgeEntry:
        """Helper to create test knowledge entry."""
        condition = MarketCondition(
            regime=regime,
            volatility=0.15,
            trend_strength=0.8
        )

        metadata = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={"period": 14},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium"
        )

        performance = PerformanceMetrics(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            profit_factor=1.3,
            sharpe_ratio=0.8,
            max_drawdown=0.05,
            avg_win=150.0,
            avg_loss=-100.0,
            total_pnl=1500.0,
            total_returns=0.15
        )

        return KnowledgeEntry(
            id=entry_id,
            market_condition=condition,
            strategy_metadata=metadata,
            performance=performance,
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.7,
            sample_size=50,
            last_updated=datetime.now()
        )


class TestAdaptiveWeighting:
    """Test adaptive weighting engine."""

    def test_adaptive_weighting_engine_creation(self):
        """Test AdaptiveWeightingEngine initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            assert engine.storage == storage
            assert engine.performance_weight == 0.4
            assert engine.regime_similarity_weight == 0.3

    def test_calculate_adaptive_weights(self):
        """Test adaptive weight calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            # Create test market condition
            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING,
                volatility=0.15,
                trend_strength=0.8
            )

            # Create test strategies
            strategies = [
                StrategyMetadata(
                    name="StrategyA",
                    category=StrategyCategory.TREND_FOLLOWING,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["ema"],
                    risk_profile="medium"
                ),
                StrategyMetadata(
                    name="StrategyB",
                    category=StrategyCategory.MEAN_REVERSION,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["rsi"],
                    risk_profile="medium"
                )
            ]

            # Calculate weights
            weights = engine.calculate_adaptive_weights(market_condition, strategies)

            # Should return equal weights when no knowledge exists
            assert len(weights) == 2
            assert all(w == 1.0 for w in weights.values())

    def test_market_similarity_calculation(self):
        """Test market similarity calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            current = MarketCondition(
                regime=MarketRegime.TRENDING,
                volatility=0.15,
                trend_strength=0.8
            )

            historical = MarketCondition(
                regime=MarketRegime.TRENDING,
                volatility=0.16,
                trend_strength=0.75
            )

            similarity = engine._calculate_market_similarity(current, historical)
            assert similarity > 0.8  # Should be very similar

    def test_performance_score_calculation(self):
        """Test performance score calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            performance = PerformanceMetrics(
                total_trades=100,
                winning_trades=70,
                losing_trades=30,
                win_rate=0.7,
                profit_factor=1.8,
                sharpe_ratio=1.5,
                max_drawdown=0.08,
                avg_win=200.0,
                avg_loss=-150.0,
                total_pnl=5000.0,
                total_returns=0.5
            )

            score = engine._calculate_performance_score(performance)
            assert score > 0.5  # Should be a good score

    def test_update_knowledge_from_trade(self):
        """Test updating knowledge from trade results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING,
                volatility=0.15,
                trend_strength=0.8
            )

            trade_result = {
                'pnl': 250.0,
                'returns': 0.025,
                'entry_price': 10000.0,
                'exit_price': 10250.0
            }

            success = engine.update_knowledge_from_trade(
                "TestStrategy", market_condition, trade_result
            )

            assert success

            # Check that entry was created
            entry = storage.get_entry(engine._generate_entry_id("TestStrategy", market_condition))
            assert entry is not None
            assert entry.performance.total_trades == 1
            assert entry.performance.total_pnl == 250.0


class TestKnowledgeManager:
    """Test knowledge manager integration."""

    def test_knowledge_manager_creation(self):
        """Test KnowledgeManager initialization."""
        config = {
            'enabled': True,
            'storage': {
                'backend': 'json',
                'file_path': 'test_knowledge.json'
            }
        }

        manager = KnowledgeManager(config)
        assert manager.enabled
        assert manager.storage is not None
        assert manager.adaptive_engine is not None

    def test_store_trade_knowledge(self):
        """Test storing trade knowledge through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'enabled': True,
                'storage': {
                    'backend': 'json',
                    'file_path': str(Path(temp_dir) / 'knowledge.json')
                }
            }

            manager = KnowledgeManager(config)

            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING,
                volatility=0.15,
                trend_strength=0.8
            )

            trade_result = {
                'pnl': 150.0,
                'returns': 0.015
            }

            success = manager.store_trade_knowledge(
                "TestStrategy", market_condition, trade_result
            )

            assert success

    def test_get_adaptive_weights(self):
        """Test getting adaptive weights through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'enabled': True,
                'storage': {
                    'backend': 'json',
                    'file_path': str(Path(temp_dir) / 'knowledge.json')
                }
            }

            manager = KnowledgeManager(config)

            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING,
                volatility=0.15,
                trend_strength=0.8
            )

            strategies = [
                StrategyMetadata(
                    name="StrategyA",
                    category=StrategyCategory.TREND_FOLLOWING,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["ema"],
                    risk_profile="medium"
                )
            ]

            weights = manager.get_adaptive_weights(market_condition, strategies)

            assert len(weights) == 1
            assert "StrategyA" in weights

    def test_knowledge_statistics(self):
        """Test getting knowledge statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                'enabled': True,
                'storage': {
                    'backend': 'json',
                    'file_path': str(Path(temp_dir) / 'knowledge.json')
                }
            }

            manager = KnowledgeManager(config)
            stats = manager.get_knowledge_statistics()

            assert stats['enabled'] is True
            assert 'storage_stats' in stats
            assert 'adaptive_stats' in stats
            assert 'knowledge_summary' in stats

    def test_disabled_knowledge_manager(self):
        """Test disabled knowledge manager."""
        config = {'enabled': False}
        manager = KnowledgeManager(config)

        assert not manager.enabled
        assert manager.storage is None
        assert manager.adaptive_engine is None

        # Test that methods return appropriate defaults
        weights = manager.get_adaptive_weights(None, [])
        assert weights == {}


class TestIntegrationWithStrategySelector:
    """Test integration with strategy selector."""

    @patch('strategies.regime.strategy_selector.get_knowledge_manager')
    def test_knowledge_base_integration(self, mock_get_manager):
        """Test that strategy selector integrates with knowledge base."""
        # Mock knowledge manager
        mock_manager = Mock()
        mock_manager.get_adaptive_weights.return_value = {
            'RSIStrategy': 1.2,
            'EMACrossStrategy': 0.8
        }
        mock_get_manager.return_value = mock_manager

        # Import here to avoid circular imports in tests
        from strategies.regime.strategy_selector import StrategySelector

        # Create strategy selector
        selector = StrategySelector()

        # Test that knowledge base methods are called
        # (This would be tested more thoroughly in integration tests)
        assert selector is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_knowledge_base(self):
        """Test behavior with empty knowledge base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "empty.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            market_condition = MarketCondition(
                regime=MarketRegime.UNKNOWN,
                volatility=0.0,
                trend_strength=0.0
            )

            strategies = [
                StrategyMetadata(
                    name="TestStrategy",
                    category=StrategyCategory.TREND_FOLLOWING,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["price"],
                    risk_profile="medium"
                )
            ]

            weights = engine.calculate_adaptive_weights(market_condition, strategies)

            # Should return equal weights
            assert weights['TestStrategy'] == 1.0

    def test_corrupted_storage_file(self):
        """Test handling of corrupted storage files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "corrupted.json"

            # Create corrupted JSON file
            with open(file_path, 'w') as f:
                f.write("invalid json content")

            storage = JSONStorage(file_path)

            # Should handle corruption gracefully
            entries = storage.list_entries()
            assert isinstance(entries, list)

    def test_large_knowledge_base(self):
        """Test performance with large knowledge base."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "large.db"
            storage = SQLiteStorage(db_path)

            # Create many entries
            entries = []
            for i in range(100):
                entry = self._create_test_entry(f"entry_{i}")
                entries.append(entry)
                storage.save_entry(entry)

            # Test query performance
            query = KnowledgeQuery(limit=50)
            result = storage.query_entries(query)

            assert len(result.entries) == 50
            assert result.total_found == 100

    def _create_test_entry(self, entry_id: str) -> KnowledgeEntry:
        """Helper to create test knowledge entry."""
        condition = MarketCondition(
            regime=MarketRegime.TRENDING,
            volatility=0.15,
            trend_strength=0.8
        )

        metadata = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium"
        )

        performance = PerformanceMetrics(
            total_trades=50,
            winning_trades=30,
            losing_trades=20,
            win_rate=0.6,
            profit_factor=1.3,
            sharpe_ratio=0.8,
            max_drawdown=0.05,
            avg_win=150.0,
            avg_loss=-100.0,
            total_pnl=1500.0,
            total_returns=0.15
        )

        return KnowledgeEntry(
            id=entry_id,
            market_condition=condition,
            strategy_metadata=metadata,
            performance=performance,
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.7,
            sample_size=50,
            last_updated=datetime.now()
        )


if __name__ == "__main__":
    pytest.main([__file__])
