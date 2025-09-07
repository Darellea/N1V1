"""
Tests for features.py - Feature extraction pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from ml.features import (
    FeatureExtractor,
    create_feature_pipeline,
    extract_features_for_symbol,
    batch_extract_features
)


class TestFeatureExtractor:
    """Test FeatureExtractor class."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        extractor = FeatureExtractor()

        assert extractor.config is not None
        assert 'indicator_params' in extractor.config
        assert 'scaling' in extractor.config
        assert 'lagged_features' in extractor.config
        assert extractor.scaler is not None
        assert extractor.feature_columns == []
        assert not extractor.is_fitted

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'indicator_params': {'rsi_period': 21},
            'scaling': {'method': 'minmax'}
        }

        extractor = FeatureExtractor(custom_config)

        assert extractor.config['indicator_params']['rsi_period'] == 21
        assert extractor.config['scaling']['method'] == 'minmax'

    @patch('ml.features.validate_ohlcv_data')
    def test_extract_features_insufficient_data(self, mock_validate):
        """Test feature extraction with insufficient data."""
        mock_validate.return_value = True

        extractor = FeatureExtractor()
        data = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [102.0] * 10,
            'volume': [1000] * 10
        })

        result = extractor.extract_features(data)

        assert result.empty

    @patch('ml.features.validate_ohlcv_data')
    @patch('ml.features.calculate_all_indicators')
    @patch('ml.features.get_indicator_names')
    def test_extract_features_success(self, mock_get_names, mock_calculate, mock_validate):
        """Test successful feature extraction."""
        mock_validate.return_value = True

        # Mock indicator calculation
        mock_data = pd.DataFrame({
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [102.0] * 100,
            'volume': [1000] * 100,
            'rsi': [50.0] * 100,
            'ema': [101.0] * 100
        })
        mock_calculate.return_value = mock_data
        mock_get_names.return_value = ['rsi', 'ema']

        extractor = FeatureExtractor()
        result = extractor.extract_features(mock_data)

        assert not result.empty
        assert 'rsi' in extractor.feature_columns
        assert 'ema' in extractor.feature_columns
        mock_calculate.assert_called_once()

    @patch('ml.features.validate_ohlcv_data')
    def test_extract_features_invalid_data(self, mock_validate):
        """Test feature extraction with invalid data."""
        mock_validate.return_value = False

        extractor = FeatureExtractor()
        data = pd.DataFrame({'invalid': [1, 2, 3]})

        with pytest.raises(ValueError, match="Data must contain OHLCV columns"):
            extractor.extract_features(data)

    def test_add_price_features(self):
        """Test adding price-based features."""
        extractor = FeatureExtractor()
        data = pd.DataFrame({
            'open': [100.0, 102.0, 104.0],
            'high': [105.0, 107.0, 109.0],
            'low': [95.0, 97.0, 99.0],
            'close': [102.0, 104.0, 106.0],
            'volume': [1000, 1100, 1200]
        })

        result = extractor._add_price_features(data)

        # Check that price features were added
        expected_features = ['returns', 'log_returns', 'price_change',
                           'high_low_ratio', 'close_open_ratio', 'body_size',
                           'upper_shadow', 'lower_shadow']

        for feature in expected_features:
            assert feature in result.columns
            assert feature in extractor.feature_columns

    def test_add_volume_features(self):
        """Test adding volume-based features."""
        extractor = FeatureExtractor()
        data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [105.0] * 50,
            'low': [95.0] * 50,
            'close': [102.0] * 50,
            'volume': list(range(1000, 1050))
        })

        result = extractor._add_volume_features(data)

        # Check that volume features were added
        expected_features = ['volume_sma_20', 'volume_ratio', 'volume_change', 'volume_ma_ratio']

        for feature in expected_features:
            assert feature in result.columns
            assert feature in extractor.feature_columns

    def test_add_lagged_features(self):
        """Test adding lagged features."""
        extractor = FeatureExtractor()
        data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [105.0] * 50,
            'low': [95.0] * 50,
            'close': list(range(100, 150)),
            'volume': [1000] * 50,
            'rsi': list(range(30, 80)),
            'ema': list(range(95, 145))
        })

        result = extractor._add_lagged_features(data)

        # Check that lagged features were added
        expected_lagged = ['close_lag_1', 'rsi_lag_1', 'ema_lag_1']

        for feature in expected_lagged:
            assert feature in result.columns
            assert feature in extractor.feature_columns

    def test_clean_features_drop_missing(self):
        """Test cleaning features with drop missing values."""
        extractor = FeatureExtractor()
        extractor.feature_columns = ['feature1', 'feature2']

        data = pd.DataFrame({
            'feature1': [1.0, 2.0, None, 4.0],
            'feature2': [5.0, 6.0, 7.0, 8.0],
            'other_col': ['a', 'b', 'c', 'd']
        })

        result = extractor._clean_features(data)

        # Should drop row with NaN
        assert len(result) == 3
        assert not result.isnull().any().any()

    def test_clean_features_fill_missing(self):
        """Test cleaning features with fill missing values."""
        config = {
            'validation': {
                'handle_missing': 'fill',
                'fill_method': 'ffill'
            }
        }
        extractor = FeatureExtractor(config)
        extractor.feature_columns = ['feature1']

        data = pd.DataFrame({
            'feature1': [1.0, None, 3.0],
            'other_col': ['a', 'b', 'c']
        })

        result = extractor._clean_features(data)

        # Should fill NaN with forward fill
        assert len(result) == 3
        assert not result['feature1'].isnull().any()

    def test_scale_features_standard_scaler(self):
        """Test scaling features with standard scaler."""
        extractor = FeatureExtractor()
        extractor.feature_columns = ['feature1', 'feature2']

        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })

        result = extractor._scale_features(data, fit_scaler=True)

        # Check that features are scaled
        assert 'feature1' in result.columns
        assert 'feature2' in result.columns
        assert 'other_col' in result.columns
        assert extractor.is_fitted

        # Check that scaled features have mean close to 0 and std close to 1
        assert abs(result['feature1'].mean()) < 0.1
        # Use population standard deviation (ddof=0) to match StandardScaler's behavior
        assert abs(np.std(result['feature1'], ddof=0) - 1.0) < 0.1

    def test_scale_features_no_scaler(self):
        """Test scaling features with no scaler configured."""
        config = {'scaling': {'method': 'none'}}
        extractor = FeatureExtractor(config)

        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'other_col': ['a', 'b', 'c']
        })

        result = extractor._scale_features(data)

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result, data)

    def test_get_feature_importance_template(self):
        """Test getting feature importance template."""
        extractor = FeatureExtractor()
        extractor.feature_columns = ['feature1', 'feature2', 'feature3']

        template = extractor.get_feature_importance_template()

        assert isinstance(template, dict)
        assert len(template) == 3
        assert all(value == 0.0 for value in template.values())

    def test_get_feature_stats(self):
        """Test getting feature statistics."""
        extractor = FeatureExtractor()
        extractor.feature_columns = ['feature1', 'feature2']

        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'other_col': ['a', 'b', 'c', 'd', 'e']
        })

        stats = extractor.get_feature_stats(data)

        assert 'feature1' in stats
        assert 'feature2' in stats

        feature1_stats = stats['feature1']
        assert 'mean' in feature1_stats
        assert 'std' in feature1_stats
        assert 'min' in feature1_stats
        assert 'max' in feature1_stats
        assert 'count' in feature1_stats

    def test_save_and_load_scaler(self):
        """Test saving and loading scaler."""
        extractor = FeatureExtractor()
        extractor.feature_columns = ['feature1']

        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        # Fit scaler first
        extractor._scale_features(data, fit_scaler=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            scaler_path = os.path.join(temp_dir, 'scaler.pkl')

            # Save scaler
            extractor.save_scaler(scaler_path)
            assert os.path.exists(scaler_path)

            # Create new extractor and load scaler
            new_extractor = FeatureExtractor()
            new_extractor.load_scaler(scaler_path)

            assert new_extractor.is_fitted
            assert new_extractor.scaler is not None


class TestModuleFunctions:
    """Test module-level functions."""

    def test_create_feature_pipeline(self):
        """Test creating feature pipeline."""
        config = {'scaling': {'method': 'minmax'}}
        extractor = create_feature_pipeline(config)

        assert isinstance(extractor, FeatureExtractor)
        assert extractor.config['scaling']['method'] == 'minmax'

    @patch('ml.features.FeatureExtractor')
    def test_extract_features_for_symbol(self, mock_extractor_class):
        """Test extracting features for a symbol."""
        mock_extractor = MagicMock()
        mock_extractor.extract_features.return_value = pd.DataFrame({'feature1': [1, 2, 3]})
        mock_extractor_class.return_value = mock_extractor

        data = pd.DataFrame({
            'open': [100.0, 102.0],
            'high': [105.0, 107.0],
            'low': [95.0, 97.0],
            'close': [102.0, 104.0],
            'volume': [1000, 1100]
        })

        symbol, features = extract_features_for_symbol(data, 'BTC/USDT')

        assert symbol == 'BTC/USDT'
        assert not features.empty
        mock_extractor.extract_features.assert_called_once_with(data)

    @patch('ml.features.FeatureExtractor')
    def test_batch_extract_features(self, mock_extractor_class):
        """Test batch feature extraction."""
        mock_extractor = MagicMock()
        mock_extractor.extract_features.return_value = pd.DataFrame({'feature1': [1, 2]})
        mock_extractor_class.return_value = mock_extractor

        data_dict = {
            'BTC/USDT': pd.DataFrame({
                'open': [100.0, 102.0],
                'high': [105.0, 107.0],
                'low': [95.0, 97.0],
                'close': [102.0, 104.0],
                'volume': [1000, 1100]
            }),
            'ETH/USDT': pd.DataFrame({
                'open': [200.0, 202.0],
                'high': [205.0, 207.0],
                'low': [195.0, 197.0],
                'close': [202.0, 204.0],
                'volume': [2000, 2100]
            })
        }

        results = batch_extract_features(data_dict)

        assert 'BTC/USDT' in results
        assert 'ETH/USDT' in results
        assert len(mock_extractor.extract_features.call_args_list) == 2

    @patch('ml.features.FeatureExtractor')
    def test_batch_extract_features_with_error(self, mock_extractor_class):
        """Test batch feature extraction with error handling."""
        mock_extractor = MagicMock()
        mock_extractor.extract_features.side_effect = [pd.DataFrame({'feature1': [1]}), Exception("Test error")]
        mock_extractor_class.return_value = mock_extractor

        data_dict = {
            'BTC/USDT': pd.DataFrame({'open': [100.0], 'high': [105.0], 'low': [95.0], 'close': [102.0], 'volume': [1000]}),
            'ETH/USDT': pd.DataFrame({'open': [200.0], 'high': [205.0], 'low': [195.0], 'close': [202.0], 'volume': [2000]})
        }

        results = batch_extract_features(data_dict)

        # Should only include successful extractions
        assert 'BTC/USDT' in results
        assert 'ETH/USDT' not in results


class TestFeatureExtractorEdgeCases:
    """Test edge cases for FeatureExtractor."""

    def test_extract_features_with_empty_data(self):
        """Test feature extraction with empty data."""
        extractor = FeatureExtractor()
        data = pd.DataFrame()

        result = extractor.extract_features(data)

        assert result.empty

    def test_clean_features_no_feature_columns(self):
        """Test cleaning features when no feature columns remain."""
        extractor = FeatureExtractor()
        extractor.feature_columns = ['nonexistent_feature']

        data = pd.DataFrame({
            'other_col': [1, 2, 3]
        })

        result = extractor._clean_features(data)

        assert result.empty

    def test_scale_features_no_feature_columns(self):
        """Test scaling features when no feature columns available."""
        extractor = FeatureExtractor()
        extractor.feature_columns = []

        data = pd.DataFrame({
            'other_col': [1, 2, 3]
        })

        result = extractor._scale_features(data)

        pd.testing.assert_frame_equal(result, data)

    def test_get_feature_stats_empty_data(self):
        """Test getting feature stats with empty data."""
        extractor = FeatureExtractor()
        extractor.feature_columns = ['feature1']

        data = pd.DataFrame({
            'feature1': [np.nan, np.nan, np.nan]
        })

        stats = extractor.get_feature_stats(data)

        assert 'feature1' not in stats or stats['feature1']['count'] == 0


if __name__ == '__main__':
    pytest.main([__file__])
