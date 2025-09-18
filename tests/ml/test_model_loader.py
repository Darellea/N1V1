"""
tests/test_model_loader.py

Comprehensive tests for ml/model_loader.py functionality.
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from ml.model_loader import load_model, load_model_with_card, predict, _align_features


class TestModelLoader:
    """Test cases for model loading functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with predict and predict_proba methods."""
        model = MagicMock()
        model.predict.return_value = np.array([1, 0, 1])
        model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        model.classes_ = np.array([0, 1])
        return model

    @pytest.fixture
    def sample_features(self):
        """Sample feature DataFrame for testing."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [0.5, 1.5, 2.5],
            'feature3': [10, 20, 30]
        })

    def test_load_model_success(self, mock_model):
        """Test successful model loading."""
        with patch('joblib.load', return_value=mock_model) as mock_joblib:
            with patch('os.path.exists', return_value=True):
                result = load_model('/path/to/model.pkl')

                assert result == mock_model
                mock_joblib.assert_called_once_with('/path/to/model.pkl')

    def test_load_model_file_not_found(self):
        """Test loading model with non-existent file."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                load_model('/nonexistent/model.pkl')

    def test_load_model_joblib_error(self):
        """Test loading model with joblib error."""
        with patch('os.path.exists', return_value=True):
            with patch('joblib.load', side_effect=Exception("Joblib error")):
                with pytest.raises(Exception, match="Joblib error"):
                    load_model('/path/to/model.pkl')

    def test_load_model_with_card_success(self, mock_model):
        """Test loading model with companion model card."""
        mock_card = {"model_type": "classifier", "version": "1.0"}

        with patch('ml.model_loader.load_model', return_value=mock_model):
            with patch('os.path.exists') as mock_exists:
                with patch('builtins.open', mock_open(read_data=json.dumps(mock_card))):
                    mock_exists.side_effect = lambda path: path.endswith('.model_card.json')

                    model, card = load_model_with_card('/path/to/model.pkl')

                    assert model == mock_model
                    assert card == mock_card

    def test_load_model_with_card_no_card_file(self, mock_model):
        """Test loading model when model card file doesn't exist."""
        with patch('ml.model_loader.load_model', return_value=mock_model):
            with patch('os.path.exists', return_value=False):
                model, card = load_model_with_card('/path/to/model.pkl')

                assert model == mock_model
                assert card is None

    def test_load_model_with_card_json_error(self, mock_model):
        """Test loading model with invalid JSON in model card."""
        with patch('ml.model_loader.load_model', return_value=mock_model):
            with patch('os.path.exists') as mock_exists:
                with patch('builtins.open', mock_open(read_data="invalid json")):
                    mock_exists.side_effect = lambda path: path.endswith('.model_card.json')

                    model, card = load_model_with_card('/path/to/model.pkl')

                    assert model == mock_model
                    assert card is None  # Should return None on JSON error

    @pytest.mark.parametrize("features_input", [
        "not a dataframe",
        None,
        [1, 2, 3],  # list instead of DataFrame
    ])
    def test_predict_invalid_features_type(self, mock_model, features_input):
        """Test predict with invalid features type."""
        with pytest.raises(ValueError, match="features must be a pandas DataFrame"):
            predict(mock_model, features_input)

    def test_predict_with_proba(self, mock_model, sample_features):
        """Test prediction with probability estimates."""
        result = predict(mock_model, sample_features)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'prediction' in result.columns
        assert 'confidence' in result.columns
        assert 'proba_0' in result.columns
        assert 'proba_1' in result.columns

        # Check predictions (match numpy array dtype)
        expected_preds = np.array([1, 0, 1])
        np.testing.assert_array_equal(result['prediction'].values, expected_preds)

        # Check probabilities
        expected_proba_0 = [0.2, 0.7, 0.1]
        expected_proba_1 = [0.8, 0.3, 0.9]
        pd.testing.assert_series_equal(result['proba_0'], pd.Series(expected_proba_0, name='proba_0'))
        pd.testing.assert_series_equal(result['proba_1'], pd.Series(expected_proba_1, name='proba_1'))

    def test_predict_without_proba(self, sample_features):
        """Test prediction without probability estimates."""
        mock_model_no_proba = MagicMock()
        mock_model_no_proba.predict.return_value = np.array([1, 0, 1])
        # No predict_proba method

        result = predict(mock_model_no_proba, sample_features)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'prediction' in result.columns
        assert 'confidence' in result.columns
        assert result['confidence'].equals(pd.Series([1.0, 1.0, 1.0], name='confidence'))

    def test_predict_model_error(self, sample_features):
        """Test prediction with model error."""
        mock_model_error = MagicMock()
        mock_model_error.predict.side_effect = Exception("Model prediction failed")

        with pytest.raises(Exception, match="Model prediction failed"):
            predict(mock_model_error, sample_features)

    def test_align_features_with_booster(self, sample_features):
        """Test feature alignment with booster feature names."""
        mock_model = MagicMock()
        mock_model.booster_ = MagicMock()
        mock_model.booster_.feature_name.return_value = ['feature2', 'feature1']

        result = _align_features(mock_model, sample_features)

        # Should be reordered to match model feature order
        expected_columns = ['feature2', 'feature1']
        assert list(result.columns) == expected_columns

    def test_align_features_with_feature_name_attr(self, sample_features):
        """Test feature alignment with feature_name_ attribute."""
        class MockModel:
            def __init__(self):
                self.feature_name_ = ['feature3', 'feature1']

        mock_model = MockModel()

        result = _align_features(mock_model, sample_features)

        expected_columns = ['feature3', 'feature1']
        assert list(result.columns) == expected_columns

    def test_align_features_missing_features(self, sample_features):
        """Test feature alignment when model expects features not in input."""
        mock_model = MagicMock()
        mock_model.booster_ = MagicMock()
        mock_model.booster_.feature_name.return_value = ['feature1', 'missing_feature', 'feature2']

        with patch('ml.model_loader.logger') as mock_logger:
            result = _align_features(mock_model, sample_features)

            # Should have added missing feature with 0.0
            assert 'missing_feature' in result.columns
            assert result['missing_feature'].equals(pd.Series([0.0, 0.0, 0.0], name='missing_feature'))

            # Check that warning was logged
            mock_logger.warning.assert_called_once()

    def test_align_features_no_feature_info(self, sample_features):
        """Test feature alignment when model has no feature information."""
        class MockModel:
            pass  # No feature-related attributes

        mock_model = MockModel()

        with patch('ml.model_loader.logger') as mock_logger:
            result = _align_features(mock_model, sample_features)

            # Should keep original order
            pd.testing.assert_frame_equal(result, sample_features)
            mock_logger.debug.assert_called_once()

    def test_align_features_booster_error(self, sample_features):
        """Test feature alignment when booster methods fail."""
        class MockBooster:
            def feature_name(self):
                raise Exception("Booster error")

        class MockModel:
            def __init__(self):
                self.booster_ = MockBooster()

        mock_model = MockModel()

        result = _align_features(mock_model, sample_features)

        # Should keep original order when booster fails
        pd.testing.assert_frame_equal(result, sample_features)

    def test_predict_proba_error_handling(self, sample_features):
        """Test prediction when predict_proba fails but predict succeeds."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        mock_model.predict_proba.side_effect = Exception("Proba failed")
        mock_model.classes_ = np.array([0, 1])

        with patch('ml.model_loader.logger') as mock_logger:
            result = predict(mock_model, sample_features)

            # Should still work with predict only
            assert 'prediction' in result.columns
            assert result['confidence'].equals(pd.Series([1.0, 1.0, 1.0], name='confidence'))
            mock_logger.warning.assert_called_once()

    def test_predict_with_classes_inference(self, sample_features):
        """Test prediction when classes_ is not available."""
        class MockModel:
            def __init__(self):
                self._predict_result = np.array([1, 0, 1])
                self._predict_proba_result = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])

            def predict(self, X):
                return self._predict_result

            def predict_proba(self, X):
                return self._predict_proba_result

        mock_model = MockModel()

        result = predict(mock_model, sample_features)

        # Should infer classes from proba shape
        assert 'proba_0' in result.columns
        assert 'proba_1' in result.columns

    def test_predict_edge_case_empty_dataframe(self):
        """Test prediction with empty DataFrame."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([])
        mock_model.predict_proba.return_value = np.array([]).reshape(0, 2)
        mock_model.classes_ = np.array([0, 1])

        empty_df = pd.DataFrame()

        result = predict(mock_model, empty_df)

        assert len(result) == 0
        assert 'prediction' in result.columns
        assert 'confidence' in result.columns

    def test_load_model_with_card_path_handling(self, mock_model):
        """Test model card path handling with different file extensions."""
        test_cases = [
            ('/path/model.pkl', '/path/model.model_card.json'),
            ('/path/model.joblib', '/path/model.model_card.json'),
            ('model.pkl', 'model.model_card.json'),
        ]

        for model_path, expected_card_path in test_cases:
            with patch('ml.model_loader.load_model', return_value=mock_model):
                with patch('os.path.exists') as mock_exists:
                    with patch('builtins.open', mock_open(read_data='{"test": "data"}')):
                        mock_exists.side_effect = lambda path: path == expected_card_path

                        model, card = load_model_with_card(model_path)

                        assert model == mock_model
                        if mock_exists.call_count > 1:  # Card file exists
                            assert card == {"test": "data"}
                        else:
                            assert card is None
