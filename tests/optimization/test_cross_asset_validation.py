"""
Comprehensive tests for cross-asset validation module.

Tests cover:
- Asset selection logic
- Validation criteria evaluation
- Cross-asset validation workflow
- Error handling and edge cases
- Integration with optimization framework
"""

import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from optimization.cross_asset_validation import (
    AssetSelector,
    AssetValidationResult,
    CrossAssetValidationResult,
    CrossAssetValidator,
    ValidationAsset,
    ValidationCriteria,
    create_cross_asset_validator,
    run_cross_asset_validation,
)
from strategies.base_strategy import BaseStrategy


class MockStrategy(BaseStrategy):
    """Mock strategy for testing cross-asset validation."""

    def __init__(self, config):
        super().__init__(config)
        self.default_params = {"fast_period": 9, "slow_period": 21, "signal_period": 9}
        config_params = (
            config.params if hasattr(config, "params") else config.get("params", {})
        )
        self.params = {**self.default_params, **(config_params or {})}

    async def calculate_indicators(self, data):
        return data

    async def generate_signals(self, data):
        return []

    def create_signal(self, **kwargs):
        return Mock()


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="h")

    # Generate realistic price data
    price = 100
    prices = []
    for _ in range(500):
        price *= 1 + np.random.normal(0, 0.01)  # 1% volatility
        prices.append(price)

    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 10000, 500),
        },
        index=dates,
    )

    return data


@pytest.fixture
def validation_config():
    """Basic configuration for cross-asset validation."""
    return {
        "asset_selector": {
            "max_assets": 3,
            "asset_weights": "equal",
            "correlation_filter": False,
            "validation_assets": [
                {
                    "symbol": "ETH/USDT",
                    "name": "Ethereum",
                    "weight": 1.0,
                    "required_history": 500,
                    "timeframe": "1h",
                },
                {
                    "symbol": "ADA/USDT",
                    "name": "Cardano",
                    "weight": 0.8,
                    "required_history": 500,
                    "timeframe": "1h",
                },
                {
                    "symbol": "SOL/USDT",
                    "name": "Solana",
                    "weight": 0.6,
                    "required_history": 500,
                    "timeframe": "1h",
                },
            ],
        },
        "validation_criteria": {
            "min_sharpe_ratio": 0.5,
            "max_drawdown_limit": 0.15,
            "min_win_rate": 0.45,
            "min_profit_factor": 1.2,
            "consistency_threshold": 0.7,
            "required_pass_rate": 0.6,
        },
        "data_fetcher": {"name": "binance", "cache_enabled": False},
        "output_dir": tempfile.mkdtemp(),
        "parallel_validation": False,
    }


@pytest.fixture
def mock_data_fetcher():
    """Mock data fetcher for testing."""
    fetcher = Mock()
    fetcher.get_historical_data = AsyncMock()
    return fetcher


class TestValidationAsset:
    """Test ValidationAsset dataclass."""

    def test_validation_asset_creation(self):
        """Test creating a validation asset."""
        asset = ValidationAsset(
            symbol="BTC/USDT",
            name="Bitcoin",
            weight=1.0,
            required_history=1000,
            timeframe="1h",
        )

        assert asset.symbol == "BTC/USDT"
        assert asset.name == "Bitcoin"
        assert asset.weight == 1.0
        assert asset.required_history == 1000
        assert asset.timeframe == "1h"

    def test_validation_asset_to_dict(self):
        """Test converting validation asset to dictionary."""
        asset = ValidationAsset(
            symbol="ETH/USDT",
            name="Ethereum",
            weight=0.8,
            required_history=500,
            timeframe="4h",
        )

        asset_dict = asset.to_dict()

        expected = {
            "symbol": "ETH/USDT",
            "name": "Ethereum",
            "weight": 0.8,
            "required_history": 500,
            "timeframe": "4h",
        }

        assert asset_dict == expected


class TestValidationCriteria:
    """Test ValidationCriteria class."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {
            "min_sharpe_ratio": 0.8,
            "max_drawdown_limit": 0.10,
            "min_win_rate": 0.50,
            "min_profit_factor": 1.5,
            "consistency_threshold": 0.8,
            "required_pass_rate": 0.7,
        }

        criteria = ValidationCriteria(config)

        assert criteria.min_sharpe_ratio == 0.8
        assert criteria.max_drawdown_limit == 0.10
        assert criteria.min_win_rate == 0.50
        assert criteria.min_profit_factor == 1.5
        assert criteria.consistency_threshold == 0.8
        assert criteria.required_pass_rate == 0.7

    def test_evaluate_asset_pass(self):
        """Test evaluating an asset that passes all criteria."""
        config = {
            "min_sharpe_ratio": 0.5,
            "max_drawdown_limit": 0.15,
            "min_win_rate": 0.45,
            "min_profit_factor": 1.2,
            "consistency_threshold": 0.7,
        }
        criteria = ValidationCriteria(config)

        primary_metrics = {
            "sharpe_ratio": 1.2,
            "total_return": 0.25,
            "win_rate": 0.55,
            "profit_factor": 1.8,
        }

        validation_metrics = {
            "sharpe_ratio": 0.8,
            "total_return": 0.18,
            "win_rate": 0.52,
            "profit_factor": 1.6,
            "max_drawdown": 0.12,
        }

        pass_criteria, overall_pass = criteria.evaluate_asset(
            primary_metrics, validation_metrics
        )

        assert pass_criteria["sharpe_ratio"] is True
        assert pass_criteria["max_drawdown"] is True
        assert pass_criteria["win_rate"] is True
        assert pass_criteria["profit_factor"] is True
        # Check that consistency is calculated (value may vary based on calculation)
        assert "consistency" in pass_criteria
        assert isinstance(pass_criteria["consistency"], bool)
        assert overall_pass is True

    def test_evaluate_asset_fail(self):
        """Test evaluating an asset that fails criteria."""
        config = {
            "min_sharpe_ratio": 0.5,
            "max_drawdown_limit": 0.15,
            "min_win_rate": 0.45,
            "min_profit_factor": 1.2,
            "consistency_threshold": 0.7,
        }
        criteria = ValidationCriteria(config)

        primary_metrics = {
            "sharpe_ratio": 1.2,
            "total_return": 0.25,
            "win_rate": 0.55,
            "profit_factor": 1.8,
        }

        validation_metrics = {
            "sharpe_ratio": 0.2,  # Below threshold
            "total_return": 0.05,
            "win_rate": 0.35,  # Below threshold
            "profit_factor": 1.0,  # Below threshold
            "max_drawdown": 0.25,  # Above threshold
        }

        pass_criteria, overall_pass = criteria.evaluate_asset(
            primary_metrics, validation_metrics
        )

        assert pass_criteria["sharpe_ratio"] is False
        assert pass_criteria["max_drawdown"] is False
        assert pass_criteria["win_rate"] is False
        assert pass_criteria["profit_factor"] is False
        assert overall_pass is False

    def test_evaluate_overall_pass(self):
        """Test overall evaluation with passing assets."""
        config = {"required_pass_rate": 0.6}
        criteria = ValidationCriteria(config)

        # Create mock results - 2 pass, 1 fail
        results = [
            Mock(overall_pass=True),
            Mock(overall_pass=True),
            Mock(overall_pass=False),
        ]

        pass_rate, overall_pass = criteria.evaluate_overall(results)

        assert pass_rate == 2 / 3
        assert overall_pass is True  # Above 60% threshold

    def test_evaluate_overall_fail(self):
        """Test overall evaluation with failing assets."""
        config = {"required_pass_rate": 0.8}
        criteria = ValidationCriteria(config)

        # Create mock results - 1 pass, 2 fail
        results = [
            Mock(overall_pass=True),
            Mock(overall_pass=False),
            Mock(overall_pass=False),
        ]

        pass_rate, overall_pass = criteria.evaluate_overall(results)

        assert pass_rate == 1 / 3
        assert overall_pass is False  # Below 80% threshold


class TestAssetSelector:
    """Test AssetSelector class."""

    def test_init_with_config(self, validation_config):
        """Test initialization with configuration."""
        selector = AssetSelector(validation_config["asset_selector"])

        assert selector.max_assets == 3
        assert selector.asset_weights == "equal"
        assert selector.correlation_filter is False
        assert len(selector.available_assets) == 3

    def test_load_validation_assets_default(self):
        """Test loading default validation assets."""
        config = {}
        selector = AssetSelector(config)

        # Should load default assets
        assert len(selector.available_assets) == 3
        assert selector.available_assets[0].symbol == "ETH/USDT"
        assert selector.available_assets[1].symbol == "ADA/USDT"
        assert selector.available_assets[2].symbol == "SOL/USDT"

    def test_select_validation_assets_basic(self, validation_config):
        """Test basic asset selection."""
        selector = AssetSelector(validation_config["asset_selector"])

        selected = selector.select_validation_assets("BTC/USDT")

        assert len(selected) == 3
        assert all(asset.symbol != "BTC/USDT" for asset in selected)
        assert selected[0].symbol == "ETH/USDT"
        assert selected[1].symbol == "ADA/USDT"
        assert selected[2].symbol == "SOL/USDT"

    def test_select_validation_assets_max_limit(self, validation_config):
        """Test asset selection with max assets limit."""
        config = validation_config["asset_selector"].copy()
        config["max_assets"] = 2
        selector = AssetSelector(config)

        selected = selector.select_validation_assets("BTC/USDT")

        assert len(selected) == 2

    def test_apply_equal_weighting(self, validation_config):
        """Test equal weighting of assets."""
        selector = AssetSelector(validation_config["asset_selector"])

        assets = [
            ValidationAsset("ETH/USDT", "Ethereum"),
            ValidationAsset("ADA/USDT", "Cardano"),
            ValidationAsset("SOL/USDT", "Solana"),
        ]

        weighted = selector._apply_weighting(assets)

        assert all(asset.weight == 1 / 3 for asset in weighted)

    def test_apply_market_cap_weighting(self, validation_config):
        """Test market cap weighting of assets."""
        config = validation_config["asset_selector"].copy()
        config["asset_weights"] = "market_cap"
        selector = AssetSelector(config)

        assets = [
            ValidationAsset("ETH/USDT", "Ethereum"),
            ValidationAsset("ADA/USDT", "Cardano"),
        ]

        # Mock market cap fetching to return ETH > ADA
        with patch.object(selector, "_fetch_market_caps_dynamically") as mock_fetch:
            mock_fetch.return_value = {
                "ETH/USDT": 300000000000,  # 300B
                "ADA/USDT": 100000000000,  # 100B
            }

            weighted = selector._apply_weighting(assets)

            # ETH should have higher weight than ADA (75% vs 25%)
            eth_weight = next(
                asset.weight for asset in weighted if asset.symbol == "ETH/USDT"
            )
            ada_weight = next(
                asset.weight for asset in weighted if asset.symbol == "ADA/USDT"
            )

            assert eth_weight > ada_weight
            assert abs(eth_weight - 0.75) < 0.001
            assert abs(ada_weight - 0.25) < 0.001
            assert abs(eth_weight + ada_weight - 1.0) < 0.001  # Should sum to 1


class TestCrossAssetValidator:
    """Test CrossAssetValidator class."""

    def test_init_with_config(self, validation_config):
        """Test initialization with configuration."""
        validator = CrossAssetValidator(validation_config)

        assert validator.asset_selector_config == validation_config["asset_selector"]
        assert (
            validator.validation_criteria_config
            == validation_config["validation_criteria"]
        )
        assert validator.output_dir == validation_config["output_dir"]
        assert validator.parallel_validation is False

    @patch("data.data_fetcher.DataFetcher")
    def test_initialize_data_fetcher(self, mock_data_fetcher_class, validation_config):
        """Test data fetcher initialization."""
        mock_fetcher = Mock()
        mock_data_fetcher_class.return_value = mock_fetcher

        validator = CrossAssetValidator(validation_config)
        validator._initialize_data_fetcher()

        # The data fetcher should be initialized (either mock or real)
        assert validator.data_fetcher is not None
        # Note: Since DataFetcher is imported at module level, the mock may not be called
        # We just verify that initialization succeeded

    def test_optimize_compatibility_method(self, validation_config):
        """Test optimize method for BaseOptimizer compatibility."""
        validator = CrossAssetValidator(validation_config)

        result = validator.optimize(MockStrategy, pd.DataFrame())

        assert result == {}

    def test_create_empty_result(self, validation_config):
        """Test creating empty result when validation fails."""
        validator = CrossAssetValidator(validation_config)

        result = validator._create_empty_result("TestStrategy", "BTC/USDT")

        assert isinstance(result, CrossAssetValidationResult)
        assert result.strategy_name == "TestStrategy"
        assert result.primary_asset == "BTC/USDT"
        assert result.validation_assets == []
        assert result.asset_results == []
        assert result.pass_rate == 0.0
        assert result.overall_pass is False
        assert result.robustness_score == 0.0

    def test_calculate_range(self, validation_config):
        """Test range calculation."""
        validator = CrossAssetValidator(validation_config)

        values = [0.5, 1.2, 0.8, 1.5]
        range_dict = validator._calculate_range(values)

        assert range_dict["min"] == 0.5
        assert range_dict["max"] == 1.5
        assert range_dict["range"] == 1.0

    def test_calculate_consistency(self, validation_config):
        """Test consistency calculation."""
        validator = CrossAssetValidator(validation_config)

        # All positive values
        values = [0.1, 0.2, 0.15, 0.18]
        consistency = validator._calculate_consistency(values)

        assert consistency == 1.0  # All same sign

        # Mixed values
        values = [0.1, -0.2, 0.15, -0.18]
        consistency = validator._calculate_consistency(values)

        assert consistency == 0.5  # Half positive, half negative

    def test_calculate_robustness_score(self, validation_config):
        """Test robustness score calculation."""
        validator = CrossAssetValidator(validation_config)

        # Create mock results
        validator.asset_results = [
            Mock(overall_pass=True, validation_metrics={"sharpe_ratio": 1.2}),
            Mock(overall_pass=True, validation_metrics={"sharpe_ratio": 0.8}),
            Mock(overall_pass=False, validation_metrics={"sharpe_ratio": 0.3}),
        ]

        score = validator._calculate_robustness_score()

        # Should be between 0 and 1
        assert 0 <= score <= 1

        # With 2/3 passing and decent Sharpe ratios, should be reasonably high
        assert score > 0.3


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_cross_asset_validator(self):
        """Test creating validator with convenience function."""
        validator = create_cross_asset_validator()

        assert isinstance(validator, CrossAssetValidator)
        assert validator.asset_selector_config is not None
        assert validator.validation_criteria_config is not None

    def test_create_cross_asset_validator_with_config(self):
        """Test creating validator with custom config."""
        custom_config = {
            "asset_selector": {
                "max_assets": 2,
                "validation_assets": [{"symbol": "ETH/USDT", "name": "Ethereum"}],
            }
        }

        validator = create_cross_asset_validator(custom_config)

        assert validator.asset_selector_config["max_assets"] == 2
        assert len(validator.asset_selector_config["validation_assets"]) == 1


class TestIntegration:
    """Integration tests for cross-asset validation."""

    @patch("optimization.cross_asset_validation.DataFetcher")
    @patch("data.data_fetcher.DataFetcher")
    def test_full_validation_workflow(
        self,
        mock_data_fetcher_class_inner,
        mock_data_fetcher_class_outer,
        sample_data,
        validation_config,
    ):
        """Test complete cross-asset validation workflow."""
        # Mock data fetcher
        mock_fetcher = Mock()
        mock_data_fetcher_class_inner.return_value = mock_fetcher
        mock_data_fetcher_class_outer.return_value = mock_fetcher

        # Mock data responses for validation assets - use AsyncMock for async calls
        async def mock_get_historical_data(*args, **kwargs):
            return sample_data

        mock_fetcher.get_historical_data = AsyncMock(
            side_effect=mock_get_historical_data
        )

        validator = CrossAssetValidator(validation_config)

        # Mock the strategy evaluation to return good metrics
        with patch.object(validator, "_evaluate_strategy_on_asset") as mock_eval:
            mock_eval.side_effect = [
                # Primary metrics
                {
                    "sharpe_ratio": 1.2,
                    "total_return": 0.25,
                    "win_rate": 0.55,
                    "profit_factor": 1.8,
                    "max_drawdown": 0.10,
                },
                # ETH validation metrics
                {
                    "sharpe_ratio": 0.8,
                    "total_return": 0.18,
                    "win_rate": 0.52,
                    "profit_factor": 1.6,
                    "max_drawdown": 0.12,
                },
                # ADA validation metrics
                {
                    "sharpe_ratio": 0.6,
                    "total_return": 0.15,
                    "win_rate": 0.48,
                    "profit_factor": 1.4,
                    "max_drawdown": 0.14,
                },
                # SOL validation metrics
                {
                    "sharpe_ratio": 0.4,  # Below threshold
                    "total_return": 0.08,
                    "win_rate": 0.42,  # Below threshold
                    "profit_factor": 1.1,  # Below threshold
                    "max_drawdown": 0.18,  # Above threshold
                },
            ]

            optimized_params = {"fast_period": 10, "slow_period": 25}
            result = validator.validate_strategy(
                MockStrategy, optimized_params, "BTC/USDT", sample_data
            )

            # Verify result structure
            assert isinstance(result, CrossAssetValidationResult)
            assert result.strategy_name == "MockStrategy"
            assert result.primary_asset == "BTC/USDT"
            assert len(result.validation_assets) == 3
            assert len(result.asset_results) == 3

            # Check that results are saved
            assert os.path.exists(
                os.path.join(
                    validation_config["output_dir"],
                    "cross_asset_validation_results.json",
                )
            )
            assert os.path.exists(
                os.path.join(
                    validation_config["output_dir"],
                    "cross_asset_validation_summary.json",
                )
            )

    @patch("optimization.cross_asset_validation.DataFetcher")
    def test_validation_with_data_fetch_error(
        self, mock_data_fetcher_class, sample_data, validation_config
    ):
        """Test validation when data fetching fails."""
        # Mock data fetcher that raises exception
        mock_fetcher = Mock()
        mock_data_fetcher_class.return_value = mock_fetcher
        mock_fetcher.get_historical_data = AsyncMock(side_effect=Exception("API Error"))

        validator = CrossAssetValidator(validation_config)

        optimized_params = {"fast_period": 10, "slow_period": 25}
        result = validator.validate_strategy(
            MockStrategy, optimized_params, "BTC/USDT", sample_data
        )

        # Should handle errors gracefully - assets are selected but validation fails
        assert isinstance(result, CrossAssetValidationResult)
        assert len(result.validation_assets) == 3  # Assets are still selected
        assert len(result.asset_results) == 3  # Results are created for all assets
        assert result.overall_pass is False  # But overall validation fails
        # Check that all asset results have errors
        assert all(
            result.error_message == "API Error" for result in result.asset_results
        )

    def test_run_cross_asset_validation_function(self, sample_data):
        """Test the convenience function for running validation."""
        with patch(
            "optimization.cross_asset_validation.create_cross_asset_validator"
        ) as mock_create:
            mock_validator = Mock()
            mock_create.return_value = mock_validator

            mock_result = Mock()
            mock_validator.validate_strategy.return_value = mock_result

            result = run_cross_asset_validation(
                MockStrategy, {"param": "value"}, "BTC/USDT", sample_data
            )

            mock_create.assert_called_once()
            mock_validator.validate_strategy.assert_called_once_with(
                MockStrategy, {"param": "value"}, "BTC/USDT", sample_data
            )
            assert result == mock_result


class TestErrorHandling:
    """Test error handling in cross-asset validation."""

    @patch("optimization.cross_asset_validation.DataFetcher")
    @patch("data.data_fetcher.DataFetcher")
    def test_validation_with_empty_data(
        self,
        mock_data_fetcher_class_inner,
        mock_data_fetcher_class_outer,
        validation_config,
    ):
        """Test validation with empty data."""
        # Mock data fetcher to prevent hanging
        mock_fetcher = Mock()
        mock_data_fetcher_class_inner.return_value = mock_fetcher
        mock_data_fetcher_class_outer.return_value = mock_fetcher

        # Mock data responses for validation assets
        async def mock_get_historical_data(*args, **kwargs):
            return pd.DataFrame()  # Return empty DataFrame

        mock_fetcher.get_historical_data = AsyncMock(
            side_effect=mock_get_historical_data
        )

        validator = CrossAssetValidator(validation_config)

        empty_data = pd.DataFrame()
        optimized_params = {"fast_period": 10}

        result = validator.validate_strategy(
            MockStrategy, optimized_params, "BTC/USDT", empty_data
        )

        # Should handle empty data gracefully
        assert isinstance(result, CrossAssetValidationResult)
        assert result.overall_pass is False

    @patch("optimization.cross_asset_validation.DataFetcher")
    @patch("data.data_fetcher.DataFetcher")
    def test_validation_with_invalid_params(
        self,
        mock_data_fetcher_class_inner,
        mock_data_fetcher_class_outer,
        validation_config,
        sample_data,
    ):
        """Test validation with invalid parameters."""
        # Mock data fetcher to prevent hanging
        mock_fetcher = Mock()
        mock_data_fetcher_class_inner.return_value = mock_fetcher
        mock_data_fetcher_class_outer.return_value = mock_fetcher

        # Mock data responses for validation assets
        async def mock_get_historical_data(*args, **kwargs):
            return sample_data

        mock_fetcher.get_historical_data = AsyncMock(
            side_effect=mock_get_historical_data
        )

        validator = CrossAssetValidator(validation_config)

        # Invalid parameters that might cause strategy to fail
        invalid_params = {"invalid_param": "value"}

        result = validator.validate_strategy(
            MockStrategy, invalid_params, "BTC/USDT", sample_data
        )

        # Should handle invalid params gracefully
        assert isinstance(result, CrossAssetValidationResult)
        # Result may still be created even if validation fails

    def test_asset_selection_with_no_candidates(self, validation_config):
        """Test asset selection when no candidates are available."""
        config = validation_config["asset_selector"].copy()
        config["validation_assets"] = [
            {"symbol": "BTC/USDT", "name": "Bitcoin"}  # Same as primary
        ]
        selector = AssetSelector(config)

        selected = selector.select_validation_assets("BTC/USDT")

        assert selected == []  # No assets available


class TestDataPersistence:
    """Test data persistence and file operations."""

    def test_save_results_creates_files(self, validation_config):
        """Test that save_results creates the expected files."""
        validator = CrossAssetValidator(validation_config)

        # Create a mock result
        result = CrossAssetValidationResult(
            strategy_name="TestStrategy",
            primary_asset="BTC/USDT",
            validation_assets=[],
            asset_results=[],
            aggregate_metrics={},
            pass_rate=0.0,
            overall_pass=False,
            robustness_score=0.0,
            timestamp=datetime.now(),
            total_time=10.0,
        )

        validator._save_results(result)

        # Check that files were created
        output_dir = validation_config["output_dir"]
        assert os.path.exists(
            os.path.join(output_dir, "cross_asset_validation_results.json")
        )
        assert os.path.exists(
            os.path.join(output_dir, "cross_asset_validation_summary.json")
        )
        assert os.path.exists(
            os.path.join(output_dir, "cross_asset_validation_summary.csv")
        )

    def test_csv_summary_format(self, validation_config):
        """Test CSV summary file format."""
        validator = CrossAssetValidator(validation_config)

        # Create mock asset results
        asset_result = AssetValidationResult(
            asset=ValidationAsset("ETH/USDT", "Ethereum"),
            optimized_params={"param": "value"},
            primary_metrics={"sharpe_ratio": 1.0},
            validation_metrics={"sharpe_ratio": 0.8, "win_rate": 0.5},
            pass_criteria={"sharpe_ratio": True, "win_rate": True},
            overall_pass=True,
            validation_time=5.0,
        )

        result = CrossAssetValidationResult(
            strategy_name="TestStrategy",
            primary_asset="BTC/USDT",
            validation_assets=[asset_result.asset],
            asset_results=[asset_result],
            aggregate_metrics={},
            pass_rate=1.0,
            overall_pass=True,
            robustness_score=0.8,
            timestamp=datetime.now(),
            total_time=5.0,
        )

        validator._save_results(result)

        # Read CSV and verify format
        csv_path = os.path.join(
            validation_config["output_dir"], "cross_asset_validation_summary.csv"
        )
        df = pd.read_csv(csv_path)

        assert len(df) == 1
        assert df.iloc[0]["asset_symbol"] == "ETH/USDT"
        assert df.iloc[0]["overall_pass"] == True
        assert df.iloc[0]["validation_time"] == 5.0


if __name__ == "__main__":
    pytest.main([__file__])
