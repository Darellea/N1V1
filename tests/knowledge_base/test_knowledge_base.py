"""
Unit tests for Knowledge Base & Adaptive Memory system.

This module contains comprehensive tests for the knowledge base functionality
including schema validation, storage operations, adaptive weighting, and
integration with strategy selection.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from knowledge_base.adaptive import (
    PERFORMANCE_WEIGHT_DEFAULT,
    RECENCY_WEIGHT_DEFAULT,
    REGIME_SIMILARITY_WEIGHT_DEFAULT,
    SAMPLE_SIZE_WEIGHT_DEFAULT,
    AdaptiveWeightingEngine,
    CacheManager,
    LRUCache,
    WeightingCalculator,
)
from knowledge_base.manager import (
    DataStoreInterface,
    KnowledgeManager,
    KnowledgeValidator,
)
from knowledge_base.schema import (
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeQueryResult,
    MarketCondition,
    MarketRegime,
    OutcomeTag,
    PerformanceMetrics,
    StrategyCategory,
    StrategyMetadata,
    validate_knowledge_entry,
)
from knowledge_base.storage import CSVStorage, JSONStorage, SQLiteStorage


class TestKnowledgeSchema:
    """Test knowledge base schema definitions."""

    def test_market_condition_creation(self):
        """Test MarketCondition dataclass creation and serialization."""
        condition = MarketCondition(
            regime=MarketRegime.TRENDING,
            volatility=0.15,
            trend_strength=0.8,
            timestamp=datetime.now(),
        )

        assert condition.regime == MarketRegime.TRENDING
        assert condition.volatility == 0.15
        assert condition.trend_strength == 0.8

        # Test serialization
        data = condition.to_dict()
        assert data["regime"] == "trending"
        assert data["volatility"] == 0.15

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
            risk_profile="medium",
        )

        assert metadata.name == "TestStrategy"
        assert metadata.category == StrategyCategory.TREND_FOLLOWING
        assert metadata.parameters == {"period": 14}

        # Test serialization
        data = metadata.to_dict()
        assert data["name"] == "TestStrategy"
        assert data["category"] == "trend_following"

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
            total_returns=0.2,
        )

        assert metrics.total_trades == 100
        assert metrics.win_rate == 0.6
        assert metrics.profit_factor == 1.5

        # Test serialization
        data = metrics.to_dict()
        assert data["total_trades"] == 100
        assert data["win_rate"] == 0.6

    def test_knowledge_entry_creation(self):
        """Test KnowledgeEntry dataclass creation and validation."""
        condition = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        metadata = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium",
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
            total_returns=0.15,
        )

        entry = KnowledgeEntry(
            id="test_entry_001",
            market_condition=condition,
            strategy_metadata=metadata,
            performance=performance,
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.8,
            sample_size=50,
            last_updated=datetime.now(),
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
            market_condition=MarketCondition(
                regime=MarketRegime.UNKNOWN, volatility=0.1, trend_strength=0.1
            ),
            strategy_metadata=StrategyMetadata(
                name="Test",
                category=StrategyCategory.TREND_FOLLOWING,
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="low",
            ),
            performance=PerformanceMetrics(
                total_trades=10,
                winning_trades=5,
                losing_trades=5,
                win_rate=0.5,
                profit_factor=1.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                total_pnl=0.0,
                total_returns=0.0,
            ),
            outcome=OutcomeTag.SUCCESS,
            confidence_score=1.5,  # Invalid: > 1.0
            sample_size=0,  # Invalid: < 1
            last_updated=datetime.now(),
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
            limit=20,
        )

        assert query.market_regime == MarketRegime.TRENDING
        assert query.strategy_name == "TestStrategy"
        assert query.min_confidence == 0.5
        assert query.limit == 20


class TestStorageBackends:
    """Test storage backend implementations."""

    def test_json_storage(self):
        """Test JSON storage backend."""
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_knowledge.json"

            storage = JSONStorage(file_path)

            # Create test entry
            entry = self._create_test_entry("test_001")

            # Test save
            assert asyncio.run(storage.save_entry(entry))

            # Test get
            retrieved = asyncio.run(storage.get_entry("test_001"))
            assert retrieved is not None
            assert retrieved.id == "test_001"

            # Test list
            entries = asyncio.run(storage.list_entries())
            assert len(entries) == 1
            assert entries[0].id == "test_001"

            # Test stats
            stats = asyncio.run(storage.get_stats())
            assert stats["backend"] == "json"
            assert stats["total_entries"] == 1

    def test_csv_storage(self):
        """Test CSV storage backend."""
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_knowledge.csv"

            storage = CSVStorage(file_path)

            # Create test entry
            entry = self._create_test_entry("test_002")

            # Test save
            assert asyncio.run(storage.save_entry(entry))

            # Test get
            retrieved = asyncio.run(storage.get_entry("test_002"))
            assert retrieved is not None
            assert retrieved.id == "test_002"

            # Test list
            entries = asyncio.run(storage.list_entries())
            assert len(entries) == 1

    def test_sqlite_storage(self):
        """Test SQLite storage backend."""
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_knowledge.db"

            storage = SQLiteStorage(db_path)

            # Create test entry
            entry = self._create_test_entry("test_003")

            # Test save
            assert asyncio.run(storage.save_entry(entry))

            # Test get
            retrieved = asyncio.run(storage.get_entry("test_003"))
            assert retrieved is not None
            assert retrieved.id == "test_003"

            # Test query
            query = KnowledgeQuery(limit=10)
            result = asyncio.run(storage.query_entries(query))
            assert result.total_found == 1
            assert len(result.entries) == 1

    def test_storage_query_filtering(self):
        """Test storage query filtering."""
        import asyncio

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
                asyncio.run(storage.save_entry(entry))

            # Test regime filtering
            query = KnowledgeQuery(market_regime=MarketRegime.TRENDING)
            result = asyncio.run(storage.query_entries(query))
            assert result.total_found == 2

            # Test confidence filtering
            query = KnowledgeQuery(min_confidence=0.9)
            result = asyncio.run(storage.query_entries(query))
            assert result.total_found == 0  # Our test entries have confidence 0.7

    def _create_test_entry(
        self, entry_id: str, regime: MarketRegime = MarketRegime.TRENDING
    ) -> KnowledgeEntry:
        """Helper to create test knowledge entry."""
        condition = MarketCondition(regime=regime, volatility=0.15, trend_strength=0.8)

        metadata = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={"period": 14},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium",
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
            total_returns=0.15,
        )

        return KnowledgeEntry(
            id=entry_id,
            market_condition=condition,
            strategy_metadata=metadata,
            performance=performance,
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.7,
            sample_size=50,
            last_updated=datetime.now(),
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
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            # Create test market condition
            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
            )

            # Create test strategies
            strategies = [
                StrategyMetadata(
                    name="StrategyA",
                    category=StrategyCategory.TREND_FOLLOWING,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["ema"],
                    risk_profile="medium",
                ),
                StrategyMetadata(
                    name="StrategyB",
                    category=StrategyCategory.MEAN_REVERSION,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["rsi"],
                    risk_profile="medium",
                ),
            ]

            # Calculate weights
            weights = asyncio.run(
                engine.calculate_adaptive_weights(market_condition, strategies)
            )

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
                regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
            )

            historical = MarketCondition(
                regime=MarketRegime.TRENDING, volatility=0.16, trend_strength=0.75
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
                total_returns=0.5,
            )

            score = engine._calculate_performance_score(performance)
            assert score > 0.5  # Should be a good score

    def test_update_knowledge_from_trade(self):
        """Test updating knowledge from trade results."""
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
            )

            trade_result = {
                "pnl": 250.0,
                "returns": 0.025,
                "entry_price": 10000.0,
                "exit_price": 10250.0,
            }

            success = asyncio.run(
                engine.update_knowledge_from_trade(
                    "TestStrategy", market_condition, trade_result
                )
            )

            assert success

            # Check that entry was created
            entry = asyncio.run(
                storage.get_entry(
                    engine._generate_entry_id("TestStrategy", market_condition)
                )
            )
            assert entry is not None
            assert entry.performance.total_trades == 1
            assert entry.performance.total_pnl == 250.0


class TestKnowledgeManager:
    """Test knowledge manager integration."""

    def test_knowledge_manager_creation(self):
        """Test KnowledgeManager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "enabled": True,
                "storage": {
                    "backend": "json",
                    "file_path": str(Path(temp_dir) / "test_knowledge.json"),
                },
            }

            manager = KnowledgeManager(config)
            assert manager.enabled
            assert manager.storage is not None
            assert manager.adaptive_engine is not None

    def test_store_trade_knowledge(self):
        """Test storing trade knowledge through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "enabled": True,
                "storage": {
                    "backend": "json",
                    "file_path": str(Path(temp_dir) / "knowledge.json"),
                },
            }

            manager = KnowledgeManager(config)

            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
            )

            trade_result = {"pnl": 150.0, "returns": 0.015}

            success = manager.store_trade_knowledge(
                "TestStrategy", market_condition, trade_result
            )

            assert success

    def test_get_adaptive_weights(self):
        """Test getting adaptive weights through manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "enabled": True,
                "storage": {
                    "backend": "json",
                    "file_path": str(Path(temp_dir) / "knowledge.json"),
                },
            }

            manager = KnowledgeManager(config)

            market_condition = MarketCondition(
                regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
            )

            strategies = [
                StrategyMetadata(
                    name="StrategyA",
                    category=StrategyCategory.TREND_FOLLOWING,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["ema"],
                    risk_profile="medium",
                )
            ]

            weights = manager.get_adaptive_weights(market_condition, strategies)

            assert len(weights) == 1
            assert "StrategyA" in weights

    def test_knowledge_statistics(self):
        """Test getting knowledge statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {
                "enabled": True,
                "storage": {
                    "backend": "json",
                    "file_path": str(Path(temp_dir) / "knowledge.json"),
                },
            }

            manager = KnowledgeManager(config)
            stats = manager.get_knowledge_statistics()

            assert stats["enabled"] is True
            assert "storage_stats" in stats
            assert "adaptive_stats" in stats
            assert "knowledge_summary" in stats

    def test_disabled_knowledge_manager(self):
        """Test disabled knowledge manager."""
        config = {"enabled": False}
        manager = KnowledgeManager(config)

        assert not manager.enabled
        assert manager.storage is None
        assert manager.adaptive_engine is None

        # Test that methods return appropriate defaults
        weights = manager.get_adaptive_weights(None, [])
        assert weights == {}


class TestIntegrationWithStrategySelector:
    """Test integration with strategy selector."""

    @patch("strategies.regime.strategy_selector.get_knowledge_manager")
    def test_knowledge_base_integration(self, mock_get_manager):
        """Test that strategy selector integrates with knowledge base."""
        # Mock knowledge manager
        mock_manager = Mock()
        mock_manager.get_adaptive_weights.return_value = {
            "RSIStrategy": 1.2,
            "EMACrossStrategy": 0.8,
        }
        mock_get_manager.return_value = mock_manager

        # Import here to avoid circular imports in tests
        from strategies.regime.strategy_selector import StrategySelector

        # Create strategy selector
        selector = StrategySelector()

        # Test that knowledge base methods are called
        # (This would be tested more thoroughly in integration tests)
        assert selector is not None


class TestWeightingCalculator:
    """Test WeightingCalculator core logic."""

    def test_calculator_initialization(self):
        """Test WeightingCalculator initialization with default config."""
        config = {}
        calculator = WeightingCalculator(config)

        assert calculator.performance_weight == PERFORMANCE_WEIGHT_DEFAULT
        assert calculator.regime_similarity_weight == REGIME_SIMILARITY_WEIGHT_DEFAULT
        assert calculator.recency_weight == RECENCY_WEIGHT_DEFAULT
        assert calculator.sample_size_weight == SAMPLE_SIZE_WEIGHT_DEFAULT

    def test_calculator_custom_config(self):
        """Test WeightingCalculator with custom config."""
        config = {
            "performance_weight": 0.5,
            "regime_similarity_weight": 0.4,
            "recency_weight": 0.05,
            "sample_size_weight": 0.05,
        }
        calculator = WeightingCalculator(config)

        assert calculator.performance_weight == 0.5
        assert calculator.regime_similarity_weight == 0.4

    def test_calculate_market_similarity_perfect_match(self):
        """Test market similarity with perfect match."""
        calculator = WeightingCalculator({})

        current = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )
        historical = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        similarity = calculator.calculate_market_similarity(current, historical)
        assert similarity == 1.0

    def test_calculate_market_similarity_no_match(self):
        """Test market similarity with no match."""
        calculator = WeightingCalculator({})

        current = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.1, trend_strength=0.9
        )
        historical = MarketCondition(
            regime=MarketRegime.SIDEWAYS, volatility=0.9, trend_strength=0.1
        )

        similarity = calculator.calculate_market_similarity(current, historical)
        assert similarity < 0.5

    def test_calculate_market_similarity_zero_volatility(self):
        """Test market similarity with zero volatility."""
        calculator = WeightingCalculator({})

        current = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.0, trend_strength=0.8
        )
        historical = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.2, trend_strength=0.8
        )

        similarity = calculator.calculate_market_similarity(current, historical)
        # Should only consider regime match since volatility is zero
        assert similarity == 1.0

    def test_calculate_recency_weight_very_recent(self):
        """Test recency weight for very recent entries."""
        calculator = WeightingCalculator({})
        recent_time = datetime.now() - timedelta(days=3)

        weight = calculator.calculate_recency_weight(recent_time)
        assert weight == 1.0  # RECENCY_MAX_WEIGHT

    def test_calculate_recency_weight_old(self):
        """Test recency weight for old entries."""
        calculator = WeightingCalculator({})
        old_time = datetime.now() - timedelta(days=120)

        weight = calculator.calculate_recency_weight(old_time)
        assert weight == 0.3  # RECENCY_MIN_WEIGHT

    def test_calculate_sample_size_weight(self):
        """Test sample size weight calculation."""
        calculator = WeightingCalculator({})

        # Small sample
        weight_small = calculator.calculate_sample_size_weight(5)
        assert weight_small < 0.5

        # Large sample
        weight_large = calculator.calculate_sample_size_weight(200)
        assert weight_large == 1.0  # Capped at 1.0

    def test_calculate_performance_score_perfect(self):
        """Test performance score with perfect metrics."""
        calculator = WeightingCalculator({})

        perf = PerformanceMetrics(
            total_trades=100,
            winning_trades=100,
            losing_trades=0,
            win_rate=1.0,
            profit_factor=5.0,
            sharpe_ratio=3.0,
            max_drawdown=0.0,
            avg_win=100.0,
            avg_loss=0.0,
            total_pnl=10000.0,
            total_returns=1.0,
        )

        score = calculator.calculate_performance_score(perf)
        assert score > 0.8

    def test_calculate_performance_score_poor(self):
        """Test performance score with poor metrics."""
        calculator = WeightingCalculator({})

        perf = PerformanceMetrics(
            total_trades=100,
            winning_trades=10,
            losing_trades=90,
            win_rate=0.1,
            profit_factor=0.5,
            sharpe_ratio=-1.0,
            max_drawdown=0.8,
            avg_win=50.0,
            avg_loss=-200.0,
            total_pnl=-5000.0,
            total_returns=-0.5,
        )

        score = calculator.calculate_performance_score(perf)
        assert score < 0.3

    def test_calculate_performance_score_none_values(self):
        """Test performance score with None values."""
        calculator = WeightingCalculator({})

        perf = PerformanceMetrics(
            total_trades=100,
            winning_trades=50,
            losing_trades=50,
            win_rate=None,  # None win_rate
            profit_factor=None,  # None profit_factor
            sharpe_ratio=None,  # None sharpe_ratio
            max_drawdown=None,  # None max_drawdown
            avg_win=100.0,
            avg_loss=-100.0,
            total_pnl=0.0,
            total_returns=0.0,
        )

        score = calculator.calculate_performance_score(perf)
        # Should handle None values gracefully
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_calculate_strategy_weight_no_knowledge(self):
        """Test strategy weight calculation with no knowledge entries."""
        calculator = WeightingCalculator({})

        strategy_meta = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium",
        )

        current_market = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        market_similarity_cache = {}

        weight = calculator.calculate_strategy_weight(
            strategy_meta, [], current_market, market_similarity_cache
        )

        assert weight == 1.0  # NO_KNOWLEDGE_DEFAULT_WEIGHT

    def test_calculate_strategy_weight_with_knowledge(self):
        """Test strategy weight calculation with knowledge entries."""
        calculator = WeightingCalculator({})

        strategy_meta = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium",
        )

        current_market = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        # Create knowledge entry
        perf = PerformanceMetrics(
            total_trades=100,
            winning_trades=70,
            losing_trades=30,
            win_rate=0.7,
            profit_factor=1.8,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            avg_win=200.0,
            avg_loss=-150.0,
            total_pnl=5000.0,
            total_returns=0.5,
        )

        entry = KnowledgeEntry(
            id="test_entry",
            market_condition=current_market,
            strategy_metadata=strategy_meta,
            performance=perf,
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.8,
            sample_size=100,
            last_updated=datetime.now() - timedelta(days=10),
        )

        knowledge_entries = [entry]
        market_similarity_cache = {current_market: 1.0}

        weight = calculator.calculate_strategy_weight(
            strategy_meta, knowledge_entries, current_market, market_similarity_cache
        )

        assert weight > 1.0  # Should boost weight for good performance

    def test_normalize_weights_empty(self):
        """Test weight normalization with empty weights."""
        calculator = WeightingCalculator({})

        normalized = calculator.normalize_weights({})
        assert normalized == {}

    def test_normalize_weights_normal(self):
        """Test weight normalization with normal weights."""
        calculator = WeightingCalculator({})

        weights = {"A": 2.0, "B": 4.0, "C": 4.0}
        normalized = calculator.normalize_weights(weights)

        total = sum(normalized.values())
        assert abs(total - 1.0) < 1e-6

        # Check relative relationships preserved
        assert normalized["B"] == normalized["C"]
        assert normalized["B"] > normalized["A"]

    def test_normalize_weights_zero_sum(self):
        """Test weight normalization with zero sum."""
        calculator = WeightingCalculator({})

        weights = {"A": 0.0, "B": 0.0, "C": 0.0}
        normalized = calculator.normalize_weights(weights)

        # Should distribute equally
        expected = 1.0 / 3
        assert all(abs(v - expected) < 1e-6 for v in normalized.values())

    def test_clamp_weight(self):
        """Test weight clamping."""
        calculator = WeightingCalculator({})

        # Test within bounds
        assert calculator._clamp_weight(1.5) == 1.5

        # Test below min
        assert calculator._clamp_weight(0.05) == 0.1

        # Test above max
        assert calculator._clamp_weight(4.0) == 3.0


class TestCacheManager:
    """Test CacheManager functionality."""

    def test_cache_manager_initialization(self):
        """Test CacheManager initialization."""
        cache_manager = CacheManager()

        assert isinstance(cache_manager._weight_cache, LRUCache)
        assert cache_manager._cache_timestamp is None
        assert cache_manager._cache_ttl == timedelta(minutes=5)

    def test_get_cache_key(self):
        """Test cache key generation."""
        cache_manager = CacheManager()

        market_condition = MarketCondition(
            regime=MarketRegime.TRENDING,
            volatility=0.15,
            trend_strength=0.8,
            timestamp=datetime(2023, 1, 1),
        )

        strategies = [
            StrategyMetadata(
                name="StrategyA",
                category=StrategyCategory.TREND_FOLLOWING,
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="medium",
            ),
            StrategyMetadata(
                name="StrategyB",
                category=StrategyCategory.MEAN_REVERSION,
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="medium",
            ),
        ]

        key = cache_manager.get_cache_key(market_condition, strategies)
        expected_key = "trending:StrategyA_StrategyB:2023-01-01"
        assert key == expected_key

    def test_cache_operations(self):
        """Test basic cache operations."""
        cache_manager = CacheManager()

        key = "test_key"
        weights = {"A": 1.2, "B": 0.8}

        # Test cache miss
        assert cache_manager.get_cached_weights(key) is None

        # Test cache set and get
        cache_manager.cache_weights(key, weights)
        cached = cache_manager.get_cached_weights(key)
        assert cached == weights

        # Test cache invalidation
        cache_manager.clear_cache()
        assert cache_manager.get_cached_weights(key) is None

    def test_cache_validity(self):
        """Test cache validity checks."""
        cache_manager = CacheManager(ttl_minutes=1)

        # Fresh cache
        cache_manager._cache_timestamp = datetime.now()
        assert cache_manager.is_cache_valid()

        # Expired cache
        cache_manager._cache_timestamp = datetime.now() - timedelta(minutes=2)
        assert not cache_manager.is_cache_valid()

        # No timestamp
        cache_manager._cache_timestamp = None
        assert not cache_manager.is_cache_valid()

    def test_get_cache_stats(self):
        """Test cache statistics."""
        cache_manager = CacheManager()

        # Empty cache
        stats = cache_manager.get_cache_stats()
        assert stats["cache_size"] == 0
        assert stats["cache_age_minutes"] is None

        # With data
        cache_manager.cache_weights("key1", {"A": 1.0})
        cache_manager.cache_weights("key2", {"B": 1.0})

        stats = cache_manager.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["cache_age_minutes"] is not None
        assert stats["cache_ttl_minutes"] == 5


class TestKnowledgeValidator:
    """Test KnowledgeValidator functionality."""

    def test_validator_initialization(self):
        """Test KnowledgeValidator initialization."""
        validator = KnowledgeValidator()
        assert "confidence_score" in validator.allowed_update_fields
        assert "sample_size" in validator.allowed_update_fields

    def test_validate_update_payload_valid(self):
        """Test validation of valid update payload."""
        validator = KnowledgeValidator()

        updates = {"confidence_score": 0.8, "sample_size": 100, "notes": "Test update"}

        errors = validator.validate_update_payload(updates)
        assert len(errors) == 0

    def test_validate_update_payload_invalid_field(self):
        """Test validation with invalid field."""
        validator = KnowledgeValidator()

        updates = {"invalid_field": "value", "confidence_score": 0.8}

        errors = validator.validate_update_payload(updates)
        assert len(errors) == 1
        assert "Unknown field" in errors[0]

    def test_validate_update_payload_invalid_type(self):
        """Test validation with invalid type."""
        validator = KnowledgeValidator()

        updates = {
            "confidence_score": "0.8",  # Should be int/float
            "sample_size": "100",  # Should be int
        }

        errors = validator.validate_update_payload(updates)
        assert len(errors) == 2
        assert any("confidence_score" in error for error in errors)
        assert any("sample_size" in error for error in errors)

    def test_validate_update_payload_invalid_values(self):
        """Test validation with invalid values."""
        validator = KnowledgeValidator()

        updates = {"confidence_score": 1.5, "sample_size": 0}  # > 1.0  # < 1

        errors = validator.validate_update_payload(updates)
        assert len(errors) == 2
        assert any("confidence" in error.lower() for error in errors)
        assert any("sample" in error.lower() for error in errors)

    def test_validate_knowledge_entry_valid(self):
        """Test validation of valid knowledge entry."""
        validator = KnowledgeValidator()

        entry = KnowledgeEntry(
            id="test_001",
            market_condition=MarketCondition(
                regime=MarketRegime.TRENDING, volatility=0.1, trend_strength=0.5
            ),
            strategy_metadata=StrategyMetadata(
                name="TestStrategy",
                category=StrategyCategory.TREND_FOLLOWING,
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="medium",
            ),
            performance=PerformanceMetrics(
                total_trades=10,
                winning_trades=5,
                losing_trades=5,
                win_rate=0.5,
                profit_factor=1.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                total_pnl=0.0,
                total_returns=0.0,
            ),
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.8,
            sample_size=10,
            last_updated=datetime.now(),
        )

        errors = validator.validate_knowledge_entry(entry)
        assert len(errors) == 0

    def test_validate_knowledge_entry_invalid(self):
        """Test validation of invalid knowledge entry."""
        validator = KnowledgeValidator()

        entry = KnowledgeEntry(
            id="",  # Empty ID
            market_condition=MarketCondition(
                regime=MarketRegime.TRENDING, volatility=0.1, trend_strength=0.5
            ),
            strategy_metadata=StrategyMetadata(
                name="",
                category=StrategyCategory.TREND_FOLLOWING,  # Empty name
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="medium",
            ),
            performance=PerformanceMetrics(
                total_trades=10,
                winning_trades=5,
                losing_trades=5,
                win_rate=0.5,
                profit_factor=1.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                total_pnl=0.0,
                total_returns=0.0,
            ),
            outcome=OutcomeTag.SUCCESS,
            confidence_score=1.2,  # > 1.0
            sample_size=0,  # < 1
            last_updated=datetime.now(),
        )

        errors = validator.validate_knowledge_entry(entry)
        assert len(errors) > 0
        assert any("ID" in error for error in errors)
        assert any("name" in error for error in errors)
        assert any("confidence" in error.lower() for error in errors)
        assert any("sample" in error.lower() for error in errors)

    def test_validate_query_parameters_valid(self):
        """Test validation of valid query parameters."""
        validator = KnowledgeValidator()

        query = KnowledgeQuery(
            market_regime=MarketRegime.TRENDING,
            min_confidence=0.5,
            max_confidence=0.9,
            min_sample_size=10,
            limit=100,
        )

        errors = validator.validate_query_parameters(query)
        assert len(errors) == 0

    def test_validate_query_parameters_invalid(self):
        """Test validation of invalid query parameters."""
        validator = KnowledgeValidator()

        query = KnowledgeQuery(
            min_confidence=0.8,
            max_confidence=0.5,  # min > max
            min_sample_size=0,  # < 1
            limit=0,  # < 1
        )

        errors = validator.validate_query_parameters(query)
        assert len(errors) == 3
        assert any("confidence" in error.lower() for error in errors)
        assert any("sample" in error.lower() for error in errors)
        assert any("limit" in error.lower() for error in errors)


class TestDataStoreInterface:
    """Test DataStoreInterface functionality."""

    def test_data_store_interface_initialization(self):
        """Test DataStoreInterface initialization."""
        mock_storage = Mock()
        data_store = DataStoreInterface(mock_storage)

        assert data_store.storage == mock_storage

    def test_save_entry(self):
        """Test save_entry method."""
        mock_storage = AsyncMock()
        data_store = DataStoreInterface(mock_storage)

        entry = Mock()
        mock_storage.save_entry.return_value = True

        result = data_store.save_entry(entry)
        assert result is True
        mock_storage.save_entry.assert_called_once_with(entry)

    def test_get_entry(self):
        """Test get_entry method."""
        mock_storage = AsyncMock()
        data_store = DataStoreInterface(mock_storage)

        mock_storage.get_entry.return_value = Mock()

        result = data_store.get_entry("test_id")
        assert result is not None
        mock_storage.get_entry.assert_called_once_with("test_id")

    def test_query_entries(self):
        """Test query_entries method."""
        mock_storage = AsyncMock()
        data_store = DataStoreInterface(mock_storage)

        query = Mock()
        mock_storage.query_entries.return_value = Mock()

        result = data_store.query_entries(query)
        assert result is not None
        mock_storage.query_entries.assert_called_once_with(query)

    def test_list_entries(self):
        """Test list_entries method."""
        mock_storage = AsyncMock()
        data_store = DataStoreInterface(mock_storage)

        mock_storage.list_entries.return_value = []

        result = data_store.list_entries(10)
        assert result == []
        mock_storage.list_entries.assert_called_once_with(10)

    def test_get_stats(self):
        """Test get_stats method."""
        mock_storage = AsyncMock()
        data_store = DataStoreInterface(mock_storage)

        mock_storage.get_stats = AsyncMock(return_value={"total": 5})

        result = data_store.get_stats()
        assert result == {"total": 5}
        mock_storage.get_stats.assert_called_once()

    def test_clear_all_success(self):
        """Test clear_all method success."""
        mock_storage = AsyncMock()
        data_store = DataStoreInterface(mock_storage)

        mock_storage.list_entries.return_value = []
        mock_storage.delete_entry.return_value = True

        result = data_store.clear_all()
        assert result is True

    def test_clear_all_with_entries(self):
        """Test clear_all method with entries."""
        mock_storage = AsyncMock()
        data_store = DataStoreInterface(mock_storage)

        # Mock entries
        mock_entries = [Mock(id="1"), Mock(id="2")]
        mock_storage.list_entries.return_value = mock_entries
        mock_storage.delete_entry.return_value = True

        result = data_store.clear_all()
        assert result is True
        assert mock_storage.delete_entry.call_count == 2


class TestAdaptiveWeightingEngine:
    """Test AdaptiveWeightingEngine functionality."""

    def test_engine_initialization(self):
        """Test AdaptiveWeightingEngine initialization."""
        mock_storage = AsyncMock()
        config = {"performance_weight": 0.5}

        engine = AdaptiveWeightingEngine(mock_storage, config)

        assert engine.storage == mock_storage
        assert engine.performance_weight == 0.5
        assert engine.calculator is not None
        assert engine.cache_manager is not None

    def test_calculate_adaptive_weights_empty_knowledge(self):
        """Test adaptive weights calculation with empty knowledge."""
        import asyncio

        mock_storage = AsyncMock()
        mock_storage.query_entries.return_value = KnowledgeQueryResult(
            [], 0, KnowledgeQuery(), 0.0
        )

        engine = AdaptiveWeightingEngine(mock_storage)

        market_condition = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        strategies = [
            StrategyMetadata(
                name="StrategyA",
                category=StrategyCategory.TREND_FOLLOWING,
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="medium",
            )
        ]

        weights = asyncio.run(
            engine.calculate_adaptive_weights(market_condition, strategies)
        )

        assert weights == {"StrategyA": 1.0}

    def test_get_strategy_recommendations(self):
        """Test strategy recommendations."""
        import asyncio

        mock_storage = AsyncMock()
        mock_storage.query_entries.return_value = KnowledgeQueryResult(
            [], 0, KnowledgeQuery(), 0.0
        )

        engine = AdaptiveWeightingEngine(mock_storage)

        market_condition = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        strategies = [
            StrategyMetadata(
                name="StrategyA",
                category=StrategyCategory.TREND_FOLLOWING,
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="medium",
            ),
            StrategyMetadata(
                name="StrategyB",
                category=StrategyCategory.MEAN_REVERSION,
                parameters={},
                timeframe="1h",
                indicators_used=[],
                risk_profile="medium",
            ),
        ]

        recommendations = asyncio.run(
            engine.get_strategy_recommendations(market_condition, strategies, top_n=1)
        )

        assert len(recommendations) == 1
        assert recommendations[0][0] in ["StrategyA", "StrategyB"]

    def test_update_knowledge_from_trade_success(self):
        """Test successful knowledge update from trade."""
        import asyncio

        mock_storage = AsyncMock()
        mock_storage.get_entry.return_value = None  # No existing entry
        mock_storage.save_entry.return_value = True

        engine = AdaptiveWeightingEngine(mock_storage)

        market_condition = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        trade_result = {
            "pnl": 250.0,
            "returns": 0.025,
            "entry_price": 10000.0,
            "exit_price": 10250.0,
        }

        success = asyncio.run(
            engine.update_knowledge_from_trade(
                "TestStrategy", market_condition, trade_result
            )
        )

        assert success
        mock_storage.save_entry.assert_called_once()

    def test_update_knowledge_from_trade_existing_entry(self):
        """Test knowledge update with existing entry."""
        import asyncio

        mock_existing_entry = Mock()
        mock_storage = AsyncMock()
        mock_storage.get_entry.return_value = mock_existing_entry
        mock_storage.save_entry.return_value = True

        engine = AdaptiveWeightingEngine(mock_storage)

        market_condition = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        trade_result = {"pnl": 150.0}

        success = asyncio.run(
            engine.update_knowledge_from_trade(
                "TestStrategy", market_condition, trade_result
            )
        )

        assert success
        mock_existing_entry.update_performance.assert_called_once()

    def test_update_knowledge_from_trade_error_handling(self):
        """Test error handling in knowledge update."""
        import asyncio

        mock_storage = AsyncMock()
        mock_storage.get_entry.side_effect = ValueError("Storage error")

        engine = AdaptiveWeightingEngine(mock_storage)

        market_condition = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        trade_result = {"pnl": 100.0}

        with pytest.raises(ValueError):
            asyncio.run(
                engine.update_knowledge_from_trade(
                    "TestStrategy", market_condition, trade_result
                )
            )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_knowledge_base(self):
        """Test behavior with empty knowledge base."""
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "empty.db"
            storage = SQLiteStorage(db_path)
            engine = AdaptiveWeightingEngine(storage)

            market_condition = MarketCondition(
                regime=MarketRegime.UNKNOWN, volatility=0.0, trend_strength=0.0
            )

            strategies = [
                StrategyMetadata(
                    name="TestStrategy",
                    category=StrategyCategory.TREND_FOLLOWING,
                    parameters={},
                    timeframe="1h",
                    indicators_used=["price"],
                    risk_profile="medium",
                )
            ]

            weights = asyncio.run(
                engine.calculate_adaptive_weights(market_condition, strategies)
            )

            # Should return equal weights
            assert weights["TestStrategy"] == 1.0

    def test_corrupted_storage_file(self):
        """Test handling of corrupted storage files."""
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "corrupted.json"

            # Create corrupted JSON file
            with open(file_path, "w") as f:
                f.write("invalid json content")

            storage = JSONStorage(file_path)

            # Should handle corruption gracefully
            entries = asyncio.run(storage.list_entries())
            assert isinstance(entries, list)

    def test_large_knowledge_base(self):
        """Test performance with large knowledge base."""
        import asyncio

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "large.db"
            storage = SQLiteStorage(db_path)

            # Create many entries
            entries = []
            for i in range(100):
                entry = self._create_test_entry(f"entry_{i}")
                entries.append(entry)
                asyncio.run(storage.save_entry(entry))

            # Test query performance
            query = KnowledgeQuery(limit=50)
            result = asyncio.run(storage.query_entries(query))

            assert len(result.entries) == 50
            assert result.total_found == 100

    def _create_test_entry(self, entry_id: str) -> KnowledgeEntry:
        """Helper to create test knowledge entry."""
        condition = MarketCondition(
            regime=MarketRegime.TRENDING, volatility=0.15, trend_strength=0.8
        )

        metadata = StrategyMetadata(
            name="TestStrategy",
            category=StrategyCategory.TREND_FOLLOWING,
            parameters={},
            timeframe="1h",
            indicators_used=["ema"],
            risk_profile="medium",
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
            total_returns=0.15,
        )

        return KnowledgeEntry(
            id=entry_id,
            market_condition=condition,
            strategy_metadata=metadata,
            performance=performance,
            outcome=OutcomeTag.SUCCESS,
            confidence_score=0.7,
            sample_size=50,
            last_updated=datetime.now(),
        )


if __name__ == "__main__":
    pytest.main([__file__])
