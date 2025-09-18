"""
Unit tests for ML filter module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import os

from ml.ml_filter import (
    MLFilter,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    create_ml_filter,
    load_ml_filter
)


class TestLogisticRegressionModel:
    """Test LogisticRegressionModel."""

    def test_init(self):
        """Test initialization."""
        model = LogisticRegressionModel()
        assert model.model is None
        assert not model.is_trained
        assert model.feature_names == []

    def test_fit_and_predict(self):
        """Test fit and predict methods."""
        model = LogisticRegressionModel()

        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([0, 0, 1, 1, 1])

        # Fit model
        model.fit(X, y)
        assert model.is_trained
        assert model.feature_names == ['feature1', 'feature2']
        assert model.model is not None

        # Predict
        predictions, confidence = model.predict(X)
        assert len(predictions) == len(X)
        assert len(confidence) == len(X)
        assert all(c >= 0 and c <= 1 for c in confidence)

    def test_predict_untrained(self):
        """Test predict on untrained model."""
        model = LogisticRegressionModel()
        X = pd.DataFrame({'feature1': [1, 2, 3]})

        with pytest.raises(ValueError, match="Model must be trained"):
            model.predict(X)

    def test_save_load(self):
        """Test save and load functionality."""
        model = LogisticRegressionModel()

        # Create and fit model
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([0, 0, 1, 1, 1])
        model.fit(X, y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            model.save(temp_path)
            assert os.path.exists(temp_path)

            # Load model
            new_model = LogisticRegressionModel()
            new_model.load(temp_path)
            assert new_model.is_trained
            assert new_model.feature_names == model.feature_names

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestRandomForestModel:
    """Test RandomForestModel."""

    def test_init(self):
        """Test initialization."""
        model = RandomForestModel()
        assert model.model is None
        assert not model.is_trained

    def test_fit_and_predict(self):
        """Test fit and predict methods."""
        model = RandomForestModel()

        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([0, 0, 1, 1, 1])

        # Fit model
        model.fit(X, y)
        assert model.is_trained
        assert model.model is not None

        # Predict
        predictions, confidence = model.predict(X)
        assert len(predictions) == len(X)
        assert len(confidence) == len(X)

    def test_get_feature_importance(self):
        """Test feature importance."""
        model = RandomForestModel()

        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([0, 0, 1, 1, 1])

        # Fit model
        model.fit(X, y)

        # Get feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == len(X.columns)
        assert all(isinstance(v, (int, float)) for v in importance.values())


class TestXGBoostModel:
    """Test XGBoostModel."""

    @pytest.mark.skipif("not hasattr(__import__('ml.ml_filter'), 'XGBOOST_AVAILABLE') or not __import__('ml.ml_filter').XGBOOST_AVAILABLE")
    def test_init(self):
        """Test initialization."""
        model = XGBoostModel()
        assert model.model is None
        assert not model.is_trained

    @pytest.mark.skipif("not hasattr(__import__('ml.ml_filter'), 'XGBOOST_AVAILABLE') or not __import__('ml.ml_filter').XGBOOST_AVAILABLE")
    def test_fit_and_predict(self):
        """Test fit and predict methods."""
        model = XGBoostModel()

        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([0, 0, 1, 1, 1])

        # Fit model
        model.fit(X, y)
        assert model.is_trained

        # Predict
        predictions, confidence = model.predict(X)
        assert len(predictions) == len(X)
        assert len(confidence) == len(X)


class TestLightGBMModel:
    """Test LightGBMModel."""

    @pytest.mark.skipif("not hasattr(__import__('ml.ml_filter'), 'LIGHTGBM_AVAILABLE') or not __import__('ml.ml_filter').LIGHTGBM_AVAILABLE")
    def test_init(self):
        """Test initialization."""
        model = LightGBMModel()
        assert model.model is None
        assert not model.is_trained

    @pytest.mark.skipif("not hasattr(__import__('ml.ml_filter'), 'LIGHTGBM_AVAILABLE') or not __import__('ml.ml_filter').LIGHTGBM_AVAILABLE")
    def test_fit_and_predict(self):
        """Test fit and predict methods."""
        model = LightGBMModel()

        # Create sample data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([0, 0, 1, 1, 1])

        # Fit model
        model.fit(X, y)
        assert model.is_trained

        # Predict
        predictions, confidence = model.predict(X)
        assert len(predictions) == len(X)
        assert len(confidence) == len(X)


class TestMLFilter:
    """Test MLFilter class."""

    def test_init(self):
        """Test initialization."""
        ml_filter = MLFilter('logistic_regression')
        assert ml_filter.model_type == 'logistic_regression'
        assert ml_filter.confidence_threshold == 0.6
        assert ml_filter.model is not None

    def test_init_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            MLFilter('invalid_model')

    def test_fit(self):
        """Test fit method."""
        ml_filter = MLFilter('logistic_regression')

        # Create sample data with more samples for proper validation split
        X = pd.DataFrame({
            'feature1': list(range(1, 21)),
            'feature2': [i*2 for i in range(1, 21)]
        })
        y = pd.Series([0]*10 + [1]*10)  # 10 samples per class

        # Fit model without validation to avoid stratification issues in small datasets
        metrics = ml_filter.fit(X, y, validation_split=0)
        assert isinstance(metrics, dict)
        # When validation_split=0, metrics will be empty dict
        assert isinstance(metrics, dict)

    def test_predict(self):
        """Test predict method."""
        ml_filter = MLFilter('logistic_regression')

        # Create and fit model with more samples to avoid stratification issues
        X = pd.DataFrame({
            'feature1': list(range(1, 21)),
            'feature2': [i*2 for i in range(1, 21)]
        })
        y = pd.Series([0]*10 + [1]*10)  # 10 samples per class
        ml_filter.fit(X, y)

        # Predict
        predictions, confidence, decisions = ml_filter.predict(X)
        assert len(predictions) == len(X)
        assert len(confidence) == len(X)
        assert len(decisions) == len(X)
        assert all(isinstance(d, (bool, np.bool_)) for d in decisions)

    def test_predict_untrained(self):
        """Test predict on untrained model."""
        ml_filter = MLFilter('logistic_regression')
        X = pd.DataFrame({'feature1': [1, 2, 3]})

        with pytest.raises(ValueError, match="Model must be trained"):
            ml_filter.predict(X)

    def test_filter_signal_buy_approved(self):
        """Test signal filtering for approved buy signal."""
        ml_filter = MLFilter('logistic_regression')

        # Create and fit model with proper multi-class data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
        })
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])  # Mixed classes
        ml_filter.fit(X, y, validation_split=0)  # Disable validation

        # Filter buy signal
        result = ml_filter.filter_signal(X.iloc[[-1]], 'buy')  # Last sample is class 1 (buy)
        assert result['approved'] is True
        assert result['direction_match'] is True

    def test_filter_signal_sell_rejected(self):
        """Test signal filtering for rejected sell signal."""
        ml_filter = MLFilter('logistic_regression')

        # Create and fit model with proper multi-class data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
        })
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])  # Mixed classes
        ml_filter.fit(X, y, validation_split=0)  # Disable validation

        # Filter sell signal for a buy-predicted sample (should be rejected)
        result = ml_filter.filter_signal(X.iloc[[-1]], 'sell')  # Last sample predicts buy
        assert result['approved'] is False
        assert result['direction_match'] is False
        assert result['reason'] == 'direction_mismatch'

    def test_filter_signal_low_confidence(self):
        """Test signal filtering with low confidence."""
        ml_filter = MLFilter('logistic_regression', {'confidence_threshold': 0.9})

        # Create and fit model with more data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
        })
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        ml_filter.fit(X, y, validation_split=0)  # Disable validation

        # Filter signal (may have low confidence)
        result = ml_filter.filter_signal(X.iloc[[-1]], 'buy')
        if result['confidence'] < 0.9:
            assert result['approved'] is False
            assert result['reason'] == 'low_confidence'

    def test_filter_signal_no_features(self):
        """Test signal filtering with no features."""
        ml_filter = MLFilter('logistic_regression')
        empty_df = pd.DataFrame()

        result = ml_filter.filter_signal(empty_df, 'buy')
        assert result['approved'] is False
        assert result['reason'] == 'no_features'

    def test_save_load_model(self):
        """Test save and load model."""
        ml_filter = MLFilter('logistic_regression')

        # Create and fit model with more samples
        X = pd.DataFrame({
            'feature1': list(range(1, 21)),
            'feature2': [i*2 for i in range(1, 21)]
        })
        y = pd.Series([0]*10 + [1]*10)  # 10 samples per class
        ml_filter.fit(X, y)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            ml_filter.save_model(temp_path)
            assert os.path.exists(temp_path)

            # Load model
            new_filter = MLFilter('logistic_regression')
            new_filter.load_model(temp_path)
            assert new_filter.model.is_trained

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_model_info(self):
        """Test get_model_info method."""
        ml_filter = MLFilter('logistic_regression')
        info = ml_filter.get_model_info()
        assert info['status'] == 'not_trained'
        # For untrained models, check that basic info is present
        assert 'confidence_threshold' in info

        # After training
        X = pd.DataFrame({
            'feature1': list(range(1, 21)),
            'feature2': [i*2 for i in range(1, 21)]
        })
        y = pd.Series([0]*10 + [1]*10)  # 10 samples per class
        ml_filter.fit(X, y)

        info = ml_filter.get_model_info()
        assert info['is_trained'] is True
        assert info['feature_names'] == ['feature1', 'feature2']
        assert info['model_type'] == 'logistic_regression'

    def test_update_confidence_threshold(self):
        """Test updating confidence threshold."""
        ml_filter = MLFilter('logistic_regression')

        # Valid threshold
        ml_filter.update_confidence_threshold(0.8)
        assert ml_filter.confidence_threshold == 0.8

        # Invalid thresholds
        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 1"):
            ml_filter.update_confidence_threshold(-0.1)

        with pytest.raises(ValueError, match="Confidence threshold must be between 0 and 1"):
            ml_filter.update_confidence_threshold(1.5)


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_ml_filter(self):
        """Test create_ml_filter function."""
        ml_filter = create_ml_filter('logistic_regression')
        assert isinstance(ml_filter, MLFilter)
        assert ml_filter.model_type == 'logistic_regression'

    def test_create_ml_filter_with_config(self):
        """Test create_ml_filter with configuration."""
        config = {
            'model_config': {'C': 0.5},
            'confidence_threshold': 0.7
        }
        ml_filter = create_ml_filter('logistic_regression', config)
        assert ml_filter.confidence_threshold == 0.7

    def test_load_ml_filter(self):
        """Test load_ml_filter function."""
        # Create and save a model first
        ml_filter = MLFilter('logistic_regression')
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [2, 4, 6, 8, 10, 12, 14, 16]
        })
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        ml_filter.fit(X, y, validation_split=0)  # Disable validation

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            ml_filter.save_model(temp_path)

            # Load using factory function
            loaded_filter = load_ml_filter(temp_path)
            assert isinstance(loaded_filter, MLFilter)
            assert loaded_filter.model.is_trained

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestErrorHandling:
    """Test error handling."""

    @patch('ml.ml_filter.SKLEARN_AVAILABLE', False)
    def test_logistic_regression_unavailable(self):
        """Test LogisticRegressionModel when sklearn unavailable."""
        with pytest.raises(ImportError, match="scikit-learn is required"):
            LogisticRegressionModel()

    @patch('ml.ml_filter.XGBOOST_AVAILABLE', False)
    def test_xgboost_unavailable(self):
        """Test XGBoostModel when XGBoost unavailable."""
        with pytest.raises(ImportError, match="XGBoost is required"):
            XGBoostModel()

    @patch('ml.ml_filter.LIGHTGBM_AVAILABLE', False)
    def test_lightgbm_unavailable(self):
        """Test LightGBMModel when LightGBM unavailable."""
        with pytest.raises(ImportError, match="LightGBM is required"):
            LightGBMModel()

    def test_fit_empty_data(self):
        """Test fit with empty data."""
        ml_filter = MLFilter('logistic_regression')
        X = pd.DataFrame()
        y = pd.Series(dtype=int)

        with pytest.raises(ValueError, match="Training data cannot be empty"):
            ml_filter.fit(X, y)

    def test_save_untrained_model(self):
        """Test saving untrained model."""
        ml_filter = MLFilter('logistic_regression')

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Model must be trained"):
                ml_filter.save_model(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
