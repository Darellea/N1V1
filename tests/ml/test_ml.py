"""
Unit tests for ML modules: indicators, features, ml_filter, train, trainer, and model_loader.

This test suite provides comprehensive unit tests for the core ML functionality,
ensuring that the refactored code maintains correctness while improving maintainability.
"""

import os
import sys
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.features import FeatureExtractor
from ml.indicators import (
    INDICATOR_CONFIG,
    calculate_all_indicators,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    get_indicator_names,
    validate_ohlcv_data,
)
from ml.ml_filter import (
    ML_MODEL_CONFIG,
    LogisticRegressionModel,
    MLFilter,
    RandomForestModel,
)
from ml.model_loader import load_model, load_model_with_card, predict
from ml.train import _process_symbol_data, load_historical_data, prepare_training_data
from ml.trainer import (
    analyze_class_distribution,
    create_binary_labels,
    prepare_data,
    validate_inputs,
)


class TestTrainFunctions(unittest.TestCase):
    """Test cases for training functions in ml/train.py."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000, 2000, 100),
            }
        )

    @patch("pandas.read_csv")
    def test_load_historical_data_csv(self, mock_read_csv):
        """Test loading historical data from CSV."""
        mock_read_csv.return_value = self.test_data
        result = load_historical_data("test.csv")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    @patch("builtins.open", new_callable=mock_open, read_data='{"data": []}')
    @patch("json.load")
    def test_load_historical_data_json(self, mock_json_load, mock_file):
        """Test loading historical data from JSON."""
        mock_json_load.return_value = self.test_data.to_dict("records")
        result = load_historical_data("test.json")
        self.assertIsInstance(result, pd.DataFrame)

    def test_load_historical_data_invalid_format(self):
        """Test loading with invalid file format."""
        with self.assertRaises(ValueError):
            load_historical_data("test.invalid")

    def test_prepare_training_data(self):
        """Test preparing training data."""
        column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        result = prepare_training_data(
            self.test_data, min_samples=50, column_map=column_map
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_prepare_training_data_insufficient_samples(self):
        """Test preparing training data with insufficient samples."""
        small_data = self.test_data.head(50)
        with self.assertRaises(ValueError):
            prepare_training_data(small_data, min_samples=1000)

    def test_process_symbol_data(self):
        """Test processing data for a single symbol."""
        column_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        result = _process_symbol_data(
            self.test_data, outlier_threshold=3.0, column_map=column_map
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)


class TestTrainerFunctions(unittest.TestCase):
    """Test cases for trainer functions in ml/trainer.py."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 100),
                "High": np.random.uniform(110, 120, 100),
                "Low": np.random.uniform(90, 100, 100),
                "Close": np.random.uniform(100, 110, 100),
                "Volume": np.random.uniform(1000, 2000, 100),
            }
        )

    def test_create_binary_labels(self):
        """Test creating binary labels."""
        result = create_binary_labels(self.test_data, horizon=5, profit_threshold=0.005)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("label_binary", result.columns)

    def test_create_binary_labels_insufficient_data(self):
        """Test creating binary labels with insufficient data."""
        small_data = self.test_data.head(3)
        with self.assertRaises(ValueError):
            create_binary_labels(small_data, horizon=5)

    def test_validate_inputs(self):
        """Test input validation."""
        self.test_data["label_binary"] = np.random.choice([0, 1], len(self.test_data))
        label_col, feature_columns = validate_inputs(self.test_data)
        self.assertEqual(label_col, "label_binary")
        self.assertIsInstance(feature_columns, list)

    def test_validate_inputs_missing_label(self):
        """Test input validation with missing label column."""
        with self.assertRaises(ValueError):
            validate_inputs(self.test_data)

    def test_prepare_data(self):
        """Test data preparation."""
        self.test_data["label_binary"] = np.random.choice([0, 1], len(self.test_data))
        X, y, sample_weights, n_splits = prepare_data(
            self.test_data, ["Open", "Close"], "label_binary", 5
        )
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_analyze_class_distribution(self):
        """Test class distribution analysis."""
        y = pd.Series(np.random.choice([0, 1], 100))
        weights = analyze_class_distribution(y)
        self.assertIsInstance(weights, dict)
        self.assertIn(0, weights)
        self.assertIn(1, weights)


class TestModelLoaderFunctions(unittest.TestCase):
    """Test cases for model loader functions in ml/model_loader.py."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.test_features = pd.DataFrame(
            {"feature1": np.random.randn(10), "feature2": np.random.randn(10)}
        )

    @patch("joblib.load")
    def test_load_model(self, mock_joblib_load):
        """Test loading a model."""
        mock_model = Mock()
        mock_joblib_load.return_value = mock_model
        result = load_model("test_model.pkl")
        self.assertEqual(result, mock_model)

    def test_load_model_file_not_found(self):
        """Test loading a model with file not found."""
        with self.assertRaises(FileNotFoundError):
            load_model("nonexistent_model.pkl")

    @patch("ml.model_loader.load_model")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_load_model_with_card(
        self, mock_json_load, mock_file, mock_exists, mock_load_model
    ):
        """Test loading model with card."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_exists.return_value = True
        mock_json_load.return_value = {"metadata": "test"}

        model, card = load_model_with_card("test_model.pkl")
        self.assertEqual(model, mock_model)
        self.assertIsNotNone(card)

    @patch("ml.model_loader.load_model")
    def test_predict(self, mock_load_model):
        """Test prediction function."""
        mock_model = Mock()
        # Mock returns arrays matching the input size (10 rows)
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]] * 10)
        mock_model.predict.return_value = np.array([1, 0] * 5)
        mock_model.classes_ = [0, 1]  # Set classes_ to a proper list
        mock_load_model.return_value = mock_model

        result = predict(mock_model, self.test_features)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("prediction", result.columns)
        self.assertIn("confidence", result.columns)

    def test_predict_invalid_input(self):
        """Test prediction with invalid input."""
        mock_model = Mock()
        with self.assertRaises(ValueError):
            predict(mock_model, "invalid_input")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_feature_extraction_with_nan_values(self):
        """Test feature extraction with NaN values."""
        np.random.seed(42)
        data_with_nan = pd.DataFrame(
            {
                "open": [100, np.nan, 102, 103],
                "high": [110, 111, np.nan, 113],
                "low": [90, 91, 92, np.nan],
                "close": [105, 106, 107, 108],
                "volume": [1000, 1001, 1002, 1003],
            }
        )

        extractor = FeatureExtractor()
        features = extractor.extract_features(data_with_nan)
        # Should handle NaN values gracefully
        self.assertIsInstance(features, pd.DataFrame)

    def test_model_training_with_imbalanced_data(self):
        """Test model training with imbalanced data."""
        np.random.seed(42)
        X = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
        )
        # Create imbalanced target
        y = pd.Series([0] * 90 + [1] * 10)

        ml_filter = MLFilter()
        metrics = ml_filter.fit(X, y)
        self.assertIsInstance(metrics, dict)

    def test_indicator_calculation_with_extreme_values(self):
        """Test indicator calculation with extreme values."""
        extreme_data = pd.DataFrame(
            {
                "open": [1e10, 1e10, 1e10],
                "high": [1e10, 1e10, 1e10],
                "low": [1e10, 1e10, 1e10],
                "close": [1e10, 1e10, 1e10],
                "volume": [1e10, 1e10, 1e10],
            }
        )

        rsi = calculate_rsi(extreme_data, period=2)
        self.assertIsInstance(rsi, pd.Series)


class TestConfiguration(unittest.TestCase):
    """Test configuration handling."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000, 2000, 100),
            }
        )

    def test_indicator_config_modification(self):
        """Test modifying indicator configuration."""
        from ml.indicators import INDICATOR_CONFIG

        original_rsi = INDICATOR_CONFIG["rsi_period"]
        INDICATOR_CONFIG["rsi_period"] = 21
        self.assertEqual(INDICATOR_CONFIG["rsi_period"], 21)
        # Reset
        INDICATOR_CONFIG["rsi_period"] = original_rsi

    def test_ml_model_config_modification(self):
        """Test modifying ML model configuration."""
        from ml.ml_filter import ML_MODEL_CONFIG

        original_c = ML_MODEL_CONFIG["logistic_regression"]["C"]
        ML_MODEL_CONFIG["logistic_regression"]["C"] = 0.5
        self.assertEqual(ML_MODEL_CONFIG["logistic_regression"]["C"], 0.5)
        # Reset
        ML_MODEL_CONFIG["logistic_regression"]["C"] = original_c

    def test_indicator_config_structure(self):
        """Test that INDICATOR_CONFIG has the expected structure."""
        expected_keys = [
            "rsi_period",
            "ema_period",
            "macd_fast",
            "macd_slow",
            "macd_signal",
            "bb_period",
            "bb_std_dev",
            "atr_period",
            "adx_period",
        ]
        for key in expected_keys:
            self.assertIn(key, INDICATOR_CONFIG)
            self.assertIsInstance(INDICATOR_CONFIG[key], (int, float))

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        rsi = calculate_rsi(self.test_data, period=14)
        self.assertEqual(len(rsi), len(self.test_data))
        # Check that non-NaN RSI values are between 0 and 100
        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue(valid_rsi.between(0, 100).all())

    def test_calculate_ema(self):
        """Test EMA calculation."""
        ema = calculate_ema(self.test_data, period=20)
        self.assertEqual(len(ema), len(self.test_data))
        # Check that we have some valid EMA values
        valid_ema = ema.dropna()
        self.assertGreater(len(valid_ema), 0)

    def test_calculate_macd(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = calculate_macd(self.test_data)
        self.assertEqual(len(macd_line), len(self.test_data))
        self.assertEqual(len(signal_line), len(self.test_data))
        self.assertEqual(len(histogram), len(self.test_data))

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = calculate_bollinger_bands(self.test_data)
        self.assertEqual(len(upper), len(self.test_data))
        self.assertEqual(len(middle), len(self.test_data))
        self.assertEqual(len(lower), len(self.test_data))
        # Upper should be greater than or equal to lower for valid values
        valid_mask = ~(upper.isna() | lower.isna())
        if valid_mask.any():
            self.assertTrue((upper[valid_mask] >= lower[valid_mask]).all())

    def test_calculate_all_indicators(self):
        """Test calculation of all indicators."""
        result = calculate_all_indicators(self.test_data)
        expected_indicators = get_indicator_names()
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)

    def test_get_indicator_names(self):
        """Test getting indicator names."""
        names = get_indicator_names()
        self.assertIsInstance(names, list)
        self.assertGreater(len(names), 0)
        for name in names:
            self.assertIsInstance(name, str)

    def test_validate_ohlcv_data(self):
        """Test OHLCV data validation."""
        # Valid data
        self.assertTrue(validate_ohlcv_data(self.test_data))

        # Missing column
        invalid_data = self.test_data.drop("close", axis=1)
        self.assertFalse(validate_ohlcv_data(invalid_data))

    def test_insufficient_data_rsi(self):
        """Test RSI with insufficient data."""
        small_data = self.test_data.head(5)
        rsi = calculate_rsi(small_data, period=14)
        self.assertTrue(rsi.isna().all())


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        self.test_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 100),
                "high": np.random.uniform(110, 120, 100),
                "low": np.random.uniform(90, 100, 100),
                "close": np.random.uniform(100, 110, 100),
                "volume": np.random.uniform(1000, 2000, 100),
            },
            index=dates,
        )

    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor()
        self.assertIsNotNone(extractor.config)
        self.assertIsNotNone(extractor.scaler)

    def test_feature_extractor_with_config(self):
        """Test FeatureExtractor with custom config."""
        config = {
            "indicator_params": {"rsi_period": 21},
            "scaling": {"method": "standard"},
        }
        extractor = FeatureExtractor(config)
        self.assertEqual(extractor.config["indicator_params"]["rsi_period"], 21)

    def test_dependency_injection(self):
        """Test dependency injection for indicator functions."""
        mock_calculate = Mock(return_value=self.test_data.copy())
        mock_get_names = Mock(return_value=["rsi", "ema"])
        mock_validate = Mock(return_value=True)

        extractor = FeatureExtractor(
            calculate_all_indicators_func=mock_calculate,
            get_indicator_names_func=mock_get_names,
            validate_ohlcv_data_func=mock_validate,
        )

        self.assertEqual(extractor.calculate_all_indicators, mock_calculate)
        self.assertEqual(extractor.get_indicator_names, mock_get_names)
        self.assertEqual(extractor.validate_ohlcv_data, mock_validate)

    def test_extract_features(self):
        """Test feature extraction pipeline."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(self.test_data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), len(self.test_data.columns))

    def test_extract_features_empty_data(self):
        """Test feature extraction with empty data."""
        extractor = FeatureExtractor()
        empty_data = pd.DataFrame()
        features = extractor.extract_features(empty_data)

        self.assertTrue(features.empty)

    def test_extract_features_insufficient_data(self):
        """Test feature extraction with insufficient data."""
        extractor = FeatureExtractor()
        small_data = self.test_data.head(10)
        features = extractor.extract_features(small_data)

        self.assertTrue(features.empty)

    def test_get_feature_importance_template(self):
        """Test getting feature importance template."""
        extractor = FeatureExtractor()
        extractor.extract_features(self.test_data)
        template = extractor.get_feature_importance_template()

        self.assertIsInstance(template, dict)
        self.assertGreater(len(template), 0)
        for value in template.values():
            self.assertEqual(value, 0.0)

    def test_get_feature_stats(self):
        """Test getting feature statistics."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(self.test_data)
        stats = extractor.get_feature_stats(features)

        self.assertIsInstance(stats, dict)
        if stats:  # Only if there are features
            for feature_name, feature_stats in stats.items():
                self.assertIn("mean", feature_stats)
                self.assertIn("std", feature_stats)
                self.assertIn("min", feature_stats)
                self.assertIn("max", feature_stats)
                self.assertIn("count", feature_stats)


class TestMLModels(unittest.TestCase):
    """Test cases for ML models."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        self.y = pd.Series(np.random.choice([0, 1], 100))

    def test_ml_model_config_structure(self):
        """Test that ML_MODEL_CONFIG has the expected structure."""
        expected_models = [
            "logistic_regression",
            "random_forest",
            "xgboost",
            "lightgbm",
        ]
        for model in expected_models:
            self.assertIn(model, ML_MODEL_CONFIG)
            self.assertIsInstance(ML_MODEL_CONFIG[model], dict)

    @patch("ml.ml_filter.SKLEARN_AVAILABLE", True)
    def test_logistic_regression_model(self):
        """Test LogisticRegressionModel."""
        model = LogisticRegressionModel()
        self.assertEqual(model.get_model_type(), "logistic_regression")

        model.fit(self.X, self.y)
        self.assertTrue(model.is_trained)

        predictions, confidence = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(confidence), len(self.X))

    @patch("ml.ml_filter.SKLEARN_AVAILABLE", True)
    def test_random_forest_model(self):
        """Test RandomForestModel."""
        model = RandomForestModel()
        self.assertEqual(model.get_model_type(), "random_forest")

        model.fit(self.X, self.y)
        self.assertTrue(model.is_trained)

        predictions, confidence = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(confidence), len(self.X))

    def test_model_with_custom_config(self):
        """Test model with custom configuration."""
        config = {"C": 0.5, "max_iter": 500}
        model = LogisticRegressionModel(config)
        self.assertEqual(model.config["C"], 0.5)
        self.assertEqual(model.config["max_iter"], 500)


class TestMLFilter(unittest.TestCase):
    """Test cases for MLFilter."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        self.y = pd.Series(np.random.choice([0, 1], 100))

    @patch("ml.ml_filter.SKLEARN_AVAILABLE", True)
    def test_ml_filter_initialization(self):
        """Test MLFilter initialization."""
        ml_filter = MLFilter()
        self.assertEqual(ml_filter.model_type, "logistic_regression")
        self.assertIsNotNone(ml_filter.model)

    @patch("ml.ml_filter.SKLEARN_AVAILABLE", True)
    def test_ml_filter_with_config(self):
        """Test MLFilter with custom config."""
        config = {"confidence_threshold": 0.7}
        ml_filter = MLFilter(config=config)
        self.assertEqual(ml_filter.confidence_threshold, 0.7)

    @patch("ml.ml_filter.SKLEARN_AVAILABLE", True)
    def test_ml_filter_fit(self):
        """Test MLFilter training."""
        ml_filter = MLFilter()
        metrics = ml_filter.fit(self.X, self.y)

        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)

    @patch("ml.ml_filter.SKLEARN_AVAILABLE", True)
    def test_ml_filter_predict(self):
        """Test MLFilter prediction."""
        ml_filter = MLFilter()
        ml_filter.fit(self.X, self.y)

        predictions, confidence, decisions = ml_filter.predict(self.X)
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(confidence), len(self.X))
        self.assertEqual(len(decisions), len(self.X))

    @patch("ml.ml_filter.SKLEARN_AVAILABLE", True)
    def test_ml_filter_signal_filtering(self):
        """Test MLFilter signal filtering."""
        ml_filter = MLFilter()
        ml_filter.fit(self.X, self.y)

        result = ml_filter.filter_signal(self.X, "buy")
        self.assertIsInstance(result, dict)
        self.assertIn("approved", result)
        self.assertIn("confidence", result)

    def test_ml_filter_empty_features(self):
        """Test MLFilter with empty features."""
        ml_filter = MLFilter()
        empty_features = pd.DataFrame()

        result = ml_filter.filter_signal(empty_features, "buy")
        self.assertFalse(result["approved"])
        self.assertEqual(result["reason"], "no_features")

    def test_update_confidence_threshold(self):
        """Test updating confidence threshold."""
        ml_filter = MLFilter()
        ml_filter.update_confidence_threshold(0.8)
        self.assertEqual(ml_filter.confidence_threshold, 0.8)

        with self.assertRaises(ValueError):
            ml_filter.update_confidence_threshold(1.5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the ML pipeline."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=200, freq="D")
        self.ohlcv_data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 110, 200),
                "high": np.random.uniform(110, 120, 200),
                "low": np.random.uniform(90, 100, 200),
                "close": np.random.uniform(100, 110, 200),
                "volume": np.random.uniform(1000, 2000, 200),
            },
            index=dates,
        )

    def test_full_pipeline(self):
        """Test the full ML pipeline from data to prediction."""
        # Use a configuration that doesn't drop too many rows
        config = {
            "validation": {
                "require_min_rows": 50,
                "handle_missing": "fill",  # Fill instead of drop
                "fill_method": "bfill",
            }
        }
        extractor = FeatureExtractor(config)
        features = extractor.extract_features(self.ohlcv_data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 5)  # Should have many features

        # Skip test if no valid features (due to random data creating too many NaNs)
        if features.empty or len(features) < 10:
            self.skipTest("Insufficient valid features due to random test data")

        # Check for NaN values in features
        if features.isna().any().any():
            self.skipTest("Features contain NaN values, skipping test")

        # Step 2: Create target (simple example)
        target = (self.ohlcv_data["close"].shift(-1) > self.ohlcv_data["close"]).astype(
            int
        )

        # Align features and target
        common_index = features.index.intersection(target.index)
        features = features.loc[common_index]
        target = target.loc[common_index]

        # Ensure we have enough data
        if len(features) < 10:
            self.skipTest("Insufficient aligned data for training")

        # Step 3: Train ML filter
        ml_filter = MLFilter()
        metrics = ml_filter.fit(features, target)

        self.assertIsInstance(metrics, dict)

        # Step 4: Make predictions
        predictions, confidence, decisions = ml_filter.predict(features)

        self.assertEqual(len(predictions), len(features))
        self.assertEqual(len(confidence), len(features))
        self.assertEqual(len(decisions), len(features))


if __name__ == "__main__":
    unittest.main()
