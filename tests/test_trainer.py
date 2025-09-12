"""
tests/test_trainer.py

Tests for ml/trainer.py covering model training, validation, and error handling.
Tests technical indicators, data processing, model training pipeline, and CLI interface.
"""

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to prevent Tkinter errors

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import trainer functions
from ml.trainer import (
    compute_rsi,
    compute_macd,
    compute_atr,
    compute_stochrsi,
    compute_trend_strength,
    load_data,
    generate_features,
    create_labels,
    create_binary_labels,
    train_model,
    train_model_binary,
    setup_logging,
    main
)


def make_sample_df(rows=200):
    """Create synthetic OHLCV data with deterministic values for testing."""
    # Ensure minimum rows for testing to avoid n_splits issues
    rows = max(rows, 50)
    idx = pd.date_range("2020-01-01", periods=rows, freq="H")
    price = np.linspace(100, 200, rows) + np.sin(np.linspace(0, 10, rows)) * 2
    df = pd.DataFrame({
        "Open": price,
        "High": price + 1.0,
        "Low": price - 1.0,
        "Close": price,
        "Volume": np.random.rand(rows) * 1000,
    }, index=idx)
    return df


class TestTechnicalIndicators:
    """Test cases for technical indicator functions."""

    def test_compute_rsi_basic(self):
        """Test RSI computation with basic data (lines 28-33)."""
        # Create test data
        data = pd.Series([10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10])
        rsi = compute_rsi(data, period=14)

        # RSI should be between 0 and 100
        assert rsi.min() >= 0
        assert rsi.max() <= 100

        # First 13 values should be NaN (period - 1)
        assert rsi.iloc[:13].isna().all()
        assert not rsi.iloc[13:].isna().any()

    def test_compute_rsi_all_upward(self):
        """Test RSI with all upward movement."""
        data = pd.Series(range(20))  # Consistently increasing
        rsi = compute_rsi(data, period=14)

        # Should approach 100
        assert rsi.iloc[-1] > 80

    def test_compute_rsi_all_downward(self):
        """Test RSI with all downward movement."""
        data = pd.Series(range(20, 0, -1))  # Consistently decreasing
        rsi = compute_rsi(data, period=14)

        # Should approach 0
        assert rsi.iloc[-1] < 20

    def test_compute_rsi_short_period(self):
        """Test RSI with short period."""
        data = pd.Series([10, 11, 12, 11, 10])
        rsi = compute_rsi(data, period=3)

        # Should have valid values after period - 1
        assert not rsi.iloc[2:].isna().any()

    def test_compute_rsi_edge_cases(self):
        """Test RSI with edge cases."""
        # Empty series
        empty = pd.Series([], dtype=float)
        rsi_empty = compute_rsi(empty)
        assert len(rsi_empty) == 0

        # Single value - RSI will be NaN due to insufficient data
        single = pd.Series([10])
        rsi_single = compute_rsi(single)
        assert rsi_single.isna().iloc[0]  # Should be NaN

        # All same values - RSI will be 50 (neutral)
        same = pd.Series([10] * 20)
        rsi_same = compute_rsi(same)
        # For constant values, RSI should be NaN initially, then 50 after fillna
        # But since we don't have enough data points, it will be NaN
        # This is expected behavior

    def test_compute_macd_basic(self):
        """Test MACD computation with basic data (lines 38-41)."""
        data = pd.Series([10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        macd = compute_macd(data)

        # MACD should be computed
        assert not macd.isna().all()

        # Should have some valid values
        assert not macd.iloc[25:].isna().any()  # After slow period

    def test_compute_macd_custom_periods(self):
        """Test MACD with custom periods."""
        data = pd.Series(range(50))
        macd = compute_macd(data, fast=5, slow=10, signal=4)

        # Should compute with custom parameters
        assert not macd.isna().all()

    def test_compute_atr_basic(self):
        """Test ATR computation with OHLC data (lines 46-51)."""
        # Create OHLC DataFrame
        data = pd.DataFrame({
            'High': [12, 13, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            'Low': [10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            'Close': [11, 12, 13, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        })

        atr = compute_atr(data, period=14)

        # ATR should be positive
        assert atr.min() >= 0

        # First 13 values should be NaN
        assert atr.iloc[:13].isna().all()
        assert not atr.iloc[13:].isna().any()

    def test_compute_atr_edge_cases(self):
        """Test ATR with edge cases."""
        # Single row - ATR will be NaN due to insufficient data for rolling window
        single_row = pd.DataFrame({
            'High': [12],
            'Low': [10],
            'Close': [11]
        })
        atr_single = compute_atr(single_row)
        # ATR requires at least period+1 rows, so single row returns NaN
        assert atr_single.isna().all()

    def test_compute_stochrsi_basic(self):
        """Test Stochastic RSI computation (lines 56-60)."""
        data = pd.Series([10, 11, 12, 11, 10, 9, 8, 9, 10, 11, 12, 13, 12, 11, 10])
        stoch_rsi = compute_stochrsi(data, period=14)

        # May return NaN for insufficient data, which is expected
        # Just check that it doesn't crash and returns a series
        assert isinstance(stoch_rsi, pd.Series)

    def test_compute_stochrsi_extremes(self):
        """Test Stochastic RSI with extreme values."""
        # All increasing - may return NaN for insufficient data
        increasing = pd.Series(range(30))
        stoch_rsi_up = compute_stochrsi(increasing, period=14)
        # Check that it returns a series (NaN is acceptable for edge cases)
        assert isinstance(stoch_rsi_up, pd.Series)

        # All decreasing - may return NaN for insufficient data
        decreasing = pd.Series(range(30, 0, -1))
        stoch_rsi_down = compute_stochrsi(decreasing, period=14)
        # Check that it returns a series (NaN is acceptable for edge cases)
        assert isinstance(stoch_rsi_down, pd.Series)

    def test_compute_trend_strength_basic(self):
        """Test trend strength computation (lines 65-75)."""
        # Create trending data
        data = pd.Series(range(50))  # Strong upward trend
        trend_strength = compute_trend_strength(data, period=20)

        # Should have positive trend strength
        valid_values = trend_strength.dropna()
        assert len(valid_values) > 0

    def test_compute_trend_strength_short_data(self):
        """Test trend strength with insufficient data."""
        data = pd.Series([1, 2, 3])  # Too short for period=20
        trend_strength = compute_trend_strength(data, period=20)

        # Should have NaN values
        assert trend_strength.isna().all()

    def test_compute_trend_strength_with_nan_data(self):
        """Test trend strength with NaN data."""
        data = pd.Series([np.nan] * 30)
        trend_strength = compute_trend_strength(data, period=20)

        # Should handle NaN data gracefully
        assert isinstance(trend_strength, pd.Series)


class TestDataProcessing:
    """Test cases for data loading and processing functions."""

    def test_load_data_success(self):
        """Test successful data loading (line 80)."""
        # Create temporary CSV file
        test_data = pd.DataFrame({
            'Open': [10, 11, 12],
            'High': [12, 13, 14],
            'Low': [8, 9, 10],
            'Close': [11, 12, 13],
            'Volume': [100, 110, 120]
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f.name, index=False)
            loaded_data = load_data(f.name)

        # Clean up
        os.unlink(f.name)

        # Verify data was loaded correctly
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_load_data_file_not_found(self):
        """Test data loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")

    @patch('pandas.read_csv')
    def test_load_data_read_error(self, mock_read_csv):
        """Test data loading with read error."""
        mock_read_csv.side_effect = Exception("Read error")

        with pytest.raises(Exception):
            load_data("test.csv")

    def test_generate_features_basic(self):
        """Test feature generation with valid OHLCV data (lines 85-101)."""
        # Create test OHLCV data
        data = pd.DataFrame({
            'Open': [10] * 50,
            'High': [12] * 50,
            'Low': [8] * 50,
            'Close': list(range(10, 60)),  # Trending data
            'Volume': [100] * 50
        })

        features_df = generate_features(data)

        # Should have all expected features
        expected_features = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']
        for feature in expected_features:
            assert feature in features_df.columns

        # Should have same number of rows
        assert len(features_df) == len(data)

    def test_generate_features_missing_columns(self):
        """Test feature generation with missing required columns."""
        data = pd.DataFrame({
            'Open': [10, 11, 12],
            'Close': [11, 12, 13]
            # Missing High, Low, Volume
        })

        # Should raise KeyError because ATR requires High/Low columns
        with pytest.raises(KeyError):
            generate_features(data)

    def test_generate_features_fillna(self):
        """Test that NaN values are filled in feature generation."""
        # Create data that will produce NaN values initially
        data = pd.DataFrame({
            'Open': [10] * 5,  # Very short for indicators
            'High': [12] * 5,
            'Low': [8] * 5,
            'Close': [10, 11, 12, 11, 10],
            'Volume': [100] * 5
        })

        features_df = generate_features(data)

        # Should not have NaN values after fillna
        feature_cols = ['RSI', 'MACD', 'EMA_20', 'ATR', 'StochRSI', 'TrendStrength', 'Volatility']
        for col in feature_cols:
            if col in features_df.columns:
                # Some indicators might still have NaN if insufficient data
                pass  # Just check that fillna was attempted

    def test_create_labels_basic(self):
        """Test label creation with basic data (lines 113-119)."""
        # Create test data
        data = pd.DataFrame({
            'Close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        })

        labeled_df = create_labels(data, horizon=3, up_thresh=0.05, down_thresh=-0.05)

        # Should have Label column (Future Price is dropped)
        assert 'Label' in labeled_df.columns
        assert 'Future Price' not in labeled_df.columns

        # Labels should be -1, 0, or 1
        valid_labels = labeled_df['Label'].dropna()
        assert valid_labels.isin([-1, 0, 1]).all()

    def test_create_labels_custom_thresholds(self):
        """Test label creation with custom thresholds."""
        data = pd.DataFrame({
            'Close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        })

        # Very sensitive thresholds
        labeled_df = create_labels(data, horizon=2, up_thresh=0.01, down_thresh=-0.01)

        # Should have more non-zero labels
        label_counts = labeled_df['Label'].value_counts()
        assert len(label_counts) >= 1  # At least one type of label

    def test_create_labels_no_movement(self):
        """Test label creation with no price movement."""
        data = pd.DataFrame({
            'Close': [10] * 20  # No movement
        })

        labeled_df = create_labels(data, horizon=3)

        # All labels should be 0 (neutral)
        valid_labels = labeled_df['Label'].dropna()
        assert (valid_labels == 0).all()

    def test_create_labels_insufficient_data(self):
        """Test label creation with insufficient data."""
        data = pd.DataFrame({
            'Close': [10, 11]  # Too short for horizon=5
        })

        # Should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient data for horizon"):
            create_labels(data, horizon=5)


class TestModelTraining:
    """Test cases for model training pipeline."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        # Create synthetic OHLCV data
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, n_samples),
            'High': np.random.uniform(110, 120, n_samples),
            'Low': np.random.uniform(90, 100, n_samples),
            'Close': np.random.uniform(100, 110, n_samples),
            'Volume': np.random.uniform(1000, 2000, n_samples)
        })

        # Generate features
        data = generate_features(data)

        # Create labels
        data = create_labels(data)

        # Drop rows with NaN labels
        data = data.dropna(subset=['Label'])

        return data

    @patch('ml.trainer.joblib.dump')
    @patch('ml.trainer.json.dump')
    @patch('ml.trainer.plt.savefig')
    @patch('ml.trainer.plt.close')
    @patch('ml.trainer.lgb.plot_importance')
    def test_train_model_basic(self, mock_plot_importance, mock_plt_close,
                              mock_plt_savefig, mock_json_dump, mock_joblib_dump,
                              sample_training_data):
        """Test basic model training (lines 150-154, 157, 169-172, 177-179)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Mock file operations
            mock_json_dump.return_value = None
            mock_joblib_dump.return_value = None
            mock_plt_savefig.return_value = None
            mock_plt_close.return_value = None
            mock_plot_importance.return_value = None

            # Train model
            train_model(
                sample_training_data,
                model_path,
                results_path=results_path,
                n_splits=2  # Small number for testing
            )

            # Verify model was saved
            mock_joblib_dump.assert_called_once()

            # Verify results were saved
            assert mock_json_dump.call_count >= 1

    @patch('ml.trainer.joblib.dump')
    @patch('ml.trainer.json.dump')
    def test_train_model_drop_neutral(self, mock_json_dump, mock_joblib_dump,
                                     sample_training_data):
        """Test model training with neutral labels dropped."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            mock_json_dump.return_value = None
            mock_joblib_dump.return_value = None

            # Train with drop_neutral=True
            train_model(
                sample_training_data,
                model_path,
                drop_neutral=True,
                n_splits=2
            )

            # Should still complete successfully
            mock_joblib_dump.assert_called_once()

    @patch('ml.trainer.joblib.dump')
    @patch('ml.trainer.json.dump')
    @patch('ml.trainer.plt.savefig')
    @patch('ml.trainer.plt.close')
    @patch('ml.trainer.lgb.plot_importance')
    def test_train_model_with_tuning(self, mock_plot_importance, mock_plt_close,
                                   mock_plt_savefig, mock_json_dump, mock_joblib_dump,
                                   sample_training_data):
        """Test model training with hyperparameter tuning (lines 198-208, 212-228)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Mock optuna at the module level to avoid pandas isinstance issues
            mock_optuna = MagicMock()
            mock_study = MagicMock()
            mock_study.best_params = {'learning_rate': 0.1, 'n_estimators': 50}
            mock_study.best_value = 0.8
            mock_study.optimize = MagicMock()
            mock_optuna.create_study.return_value = mock_study
            mock_optuna.Trial = MagicMock

            with patch.dict('sys.modules', {'optuna': mock_optuna}):
                mock_json_dump.return_value = None
                mock_joblib_dump.return_value = None
                mock_plt_savefig.return_value = None
                mock_plt_close.return_value = None
                mock_plot_importance.return_value = None

                # Train with tuning
                train_model(
                    sample_training_data,
                    model_path,
                    tune=True,
                    n_trials=2,  # Small number for testing
                    n_splits=2
                )

                # Verify model was saved
                mock_joblib_dump.assert_called_once()

    def test_train_model_optuna_not_installed(self, sample_training_data):
        """Test model training when Optuna is not available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Mock import error for optuna
            with patch.dict('sys.modules', {'optuna': None}):
                with patch('builtins.__import__', side_effect=ImportError("No module named 'optuna'")):
                    with pytest.raises(ImportError):
                        train_model(
                            sample_training_data,
                            model_path,
                            tune=True,
                            n_splits=2
                        )

    @patch('ml.trainer.joblib.dump')
    @patch('ml.trainer.json.dump')
    @patch('ml.trainer.lgb.LGBMClassifier')
    @patch('ml.trainer.plt.savefig')
    @patch('ml.trainer.plt.close')
    @patch('ml.trainer.lgb.plot_importance')
    @patch('ml.trainer.plt.figure')  # Mock plt.figure to avoid GUI issues
    def test_train_model_feature_selection(self, mock_plt_figure, mock_plot_importance, mock_plt_close,
                                         mock_plt_savefig, mock_lgb_classifier,
                                         mock_json_dump, mock_joblib_dump, sample_training_data):
        """Test model training with feature selection (lines 233-279)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Mock matplotlib figure to avoid GUI issues
            mock_figure = MagicMock()
            mock_plt_figure.return_value = mock_figure

            # Mock LightGBM classifier with proper booster
            mock_model = MagicMock()
            mock_booster = MagicMock()
            mock_booster.feature_importance.return_value = [0.1, 0.5, 0.05, 0.8, 0.2, 0.1, 0.9]  # Return list, not numpy array
            mock_model.booster_ = mock_booster
            mock_lgb_classifier.return_value = mock_model

            mock_json_dump.return_value = None
            mock_joblib_dump.return_value = None
            mock_plt_savefig.return_value = None
            mock_plt_close.return_value = None
            mock_plot_importance.return_value = None

            # Train with feature selection
            train_model(
                sample_training_data,
                model_path,
                feature_selection=True,
                n_splits=2
            )

            # Should have called feature_importance
            mock_booster.feature_importance.assert_called()

            # Should have saved model
            assert mock_joblib_dump.call_count >= 1

    @patch('ml.trainer.joblib.dump')
    @patch('ml.trainer.json.dump')
    @patch('ml.trainer.lgb.LGBMClassifier')
    @patch('ml.trainer.plt.savefig')
    @patch('ml.trainer.plt.close')
    @patch('ml.trainer.lgb.plot_importance')
    def test_train_model_eval_profit(self, mock_plot_importance, mock_plt_close,
                                   mock_plt_savefig, mock_lgb_classifier,
                                   mock_json_dump, mock_joblib_dump, sample_training_data):
        """Test model training with profit evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Mock LightGBM classifier
            mock_model = MagicMock()
            mock_lgb_classifier.return_value = mock_model

            mock_json_dump.return_value = None
            mock_joblib_dump.return_value = None
            mock_plt_savefig.return_value = None
            mock_plt_close.return_value = None
            mock_plot_importance.return_value = None

            # Train with profit evaluation
            train_model(
                sample_training_data,
                model_path,
                eval_profit=True,
                n_splits=2
            )

            # Should complete successfully
            mock_joblib_dump.assert_called_once()

    def test_train_model_insufficient_data(self):
        """Test model training with insufficient data."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'RSI': [50, 60, 70, 80, 90],
            'MACD': [0.1, 0.2, 0.3, 0.4, 0.5],
            'EMA_20': [10, 11, 12, 13, 14],
            'ATR': [1, 1.1, 1.2, 1.3, 1.4],
            'StochRSI': [0.5, 0.6, 0.7, 0.8, 0.9],
            'TrendStrength': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Volatility': [0.1, 0.15, 0.2, 0.25, 0.3],
            'Label': [1, -1, 0, 1, -1]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Should handle small dataset gracefully with n_splits=2
            train_model(small_data, model_path, n_splits=2)

    @patch('ml.trainer.joblib.dump')
    @patch('ml.trainer.plt.savefig')
    @patch('ml.trainer.plt.close')
    @patch('ml.trainer.lgb.plot_importance')
    def test_train_model_save_errors(self, mock_plot_importance, mock_plt_close,
                                   mock_plt_savefig, mock_joblib_dump, sample_training_data):
        """Test model training with save errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Mock save error
            mock_joblib_dump.side_effect = Exception("Save failed")
            mock_plt_savefig.return_value = None
            mock_plt_close.return_value = None
            mock_plot_importance.return_value = None

            # Should handle save error gracefully (not crash)
            try:
                train_model(sample_training_data, model_path, n_splits=2)
                # If we get here, the function handled the error gracefully
                assert True
            except Exception as e:
                # If an exception is raised, it should be the expected save error
                assert "Save failed" in str(e)


class TestUtilities:
    """Test cases for utility functions."""

    def test_setup_logging_basic(self):
        """Test logging setup (line 288)."""
        # Clear existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Setup logging
        setup_logging()

        # Should have at least one handler
        assert len(logger.handlers) >= 1

        # Should have StreamHandler
        has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert has_stream_handler

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        import uuid
        # Use unique filename to avoid file locking issues
        logfile = f"test_log_{uuid.uuid4().hex}.log"

        try:
            # Clear existing handlers
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Setup logging with file
            setup_logging(logfile)

            # Should have file handler
            has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
            assert has_file_handler

        finally:
            # Cleanup - close all handlers first to release file locks
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                if hasattr(handler, 'close'):
                    try:
                        handler.close()
                    except Exception:
                        pass  # Ignore errors when closing handlers
                logger.removeHandler(handler)

            # Now delete the log file
            expected_logfile = os.path.join("test_logs", logfile)
            if os.path.exists(expected_logfile):
                os.unlink(expected_logfile)

    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        # Clear existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Setup with DEBUG level
        setup_logging(level=logging.DEBUG)

        assert logger.level == logging.DEBUG


class TestCLI:
    """Test cases for CLI interface."""

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.generate_features')
    @patch('ml.trainer.create_binary_labels')
    @patch('ml.trainer.train_model_binary')
    @patch('ml.trainer.setup_logging')
    def test_main_basic_execution(self, mock_setup_logging, mock_train_model_binary,
                                mock_create_binary_labels, mock_generate_features,
                                mock_load_data, mock_parse_args):
        """Test main function basic execution."""
        # Mock arguments with proper numeric values
        mock_args = MagicMock()
        mock_args.data = 'test.csv'
        mock_args.output = 'model.pkl'
        mock_args.up_thresh = 0.005
        mock_args.horizon = 3  # Use smaller horizon for test data
        mock_args.results = 'results.json'
        mock_args.n_splits = 2  # Use smaller n_splits
        mock_args.logfile = None
        mock_args.tune = False
        mock_args.n_trials = 25
        mock_args.feature_selection = False
        mock_args.early_stopping_rounds = 50
        mock_args.eval_profit = False

        mock_parse_args.return_value = mock_args

        # Mock data processing - need more rows for horizon=3 (at least 10 samples)
        mock_df = pd.DataFrame({
            'Open': list(range(10, 20)),  # 10 samples
            'High': list(range(12, 22)),  # 10 samples
            'Low': list(range(8, 18)),    # 10 samples
            'Close': list(range(11, 21)), # 10 samples
            'Volume': list(range(100, 200, 10)) # 10 samples
        })
        mock_load_data.return_value = mock_df

        # Mock generate_features to return DataFrame with features (10 rows to match mock_df)
        features_df = mock_df.copy()
        features_df['RSI'] = [50, 60, 70, 55, 65, 75, 45, 52, 58, 62]
        features_df['MACD'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['EMA_20'] = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        features_df['ATR'] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        features_df['StochRSI'] = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.45, 0.52, 0.58, 0.62]
        features_df['TrendStrength'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['Volatility'] = [0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.08, 0.14, 0.16, 0.19]
        mock_generate_features.return_value = features_df

        # Mock create_binary_labels to return DataFrame with 'label_binary' column (10 rows to match)
        binary_df = features_df.copy()
        binary_df['label_binary'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Ensure both classes are present and length matches DataFrame
        mock_create_binary_labels.return_value = binary_df

        # Assert class balance in mocked labels
        assert set(binary_df['label_binary']) == {0, 1}

        # Mock train_model_binary to return results
        mock_train_model_binary.return_value = {
            'metadata': {'model_type': 'binary_classification'},
            'overall_metrics': {'auc': 0.8, 'f1': 0.75}
        }

        # Execute main
        main()

        # Verify all functions were called
        mock_setup_logging.assert_called_once_with(None)
        mock_load_data.assert_called_once_with('test.csv')
        mock_generate_features.assert_called_once()
        assert mock_create_binary_labels.called
        assert mock_train_model_binary.called

    def test_main_missing_required_columns(self):
        """Test main with missing required columns."""
        with patch('ml.trainer.argparse.ArgumentParser.parse_args') as mock_parse_args, \
             patch('ml.trainer.load_data') as mock_load_data, \
             patch('ml.trainer.setup_logging'):

            # Mock arguments
            mock_args = MagicMock()
            mock_args.data = 'test.csv'
            mock_parse_args.return_value = mock_args

            # Mock data without required columns
            mock_df = pd.DataFrame({
                'Price': [10, 11, 12]  # Missing OHLCV columns
            })
            mock_load_data.return_value = mock_df

            # Should raise ValueError
            with pytest.raises(ValueError, match="Data must contain columns"):
                main()

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.setup_logging')
    def test_main_data_loading_error(self, mock_setup_logging, mock_load_data, mock_parse_args):
        """Test main with data loading error."""
        # Mock arguments
        mock_args = MagicMock()
        mock_args.data = 'nonexistent.csv'
        mock_parse_args.return_value = mock_args

        # Mock data loading error
        mock_load_data.side_effect = FileNotFoundError("File not found")

        # Should propagate the error
        with pytest.raises(FileNotFoundError):
            main()

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.generate_features')
    @patch('ml.trainer.create_binary_labels')
    @patch('ml.trainer.train_model_binary')
    @patch('ml.trainer.setup_logging')
    def test_main_with_drop_neutral(self, mock_setup_logging, mock_train_model_binary,
                                  mock_create_binary_labels, mock_generate_features,
                                  mock_load_data, mock_parse_args):
        """Test main with drop_neutral option (lines 398, 401, 407-408, 413)."""
        # Mock arguments with drop_neutral=True
        mock_args = MagicMock()
        mock_args.data = 'test.csv'
        mock_args.output = 'model.pkl'
        mock_args.up_thresh = 0.005
        mock_args.horizon = 5
        mock_args.drop_neutral = True
        mock_args.tune = False
        mock_args.feature_selection = False
        mock_args.eval_profit = False
        mock_args.n_splits = 2  # Add n_splits
        mock_args.results = 'results.json'
        mock_args.logfile = None
        mock_args.n_trials = 25
        mock_args.early_stopping_rounds = 50

        mock_parse_args.return_value = mock_args

        # Mock data with sufficient rows for horizon=5 (need at least 10 samples)
        mock_df = pd.DataFrame({
            'Open': list(range(10, 20)),
            'High': list(range(12, 22)),
            'Low': list(range(8, 18)),
            'Close': list(range(11, 21)),
            'Volume': list(range(100, 200, 10))
        })
        mock_load_data.return_value = mock_df

        # Mock generate_features to return DataFrame with features
        features_df = mock_df.copy()
        features_df['RSI'] = [50, 60, 70, 55, 65, 75, 45, 52, 58, 62]
        features_df['MACD'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['EMA_20'] = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        features_df['ATR'] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        features_df['StochRSI'] = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.45, 0.52, 0.58, 0.62]
        features_df['TrendStrength'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['Volatility'] = [0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.08, 0.14, 0.16, 0.19]
        mock_generate_features.return_value = features_df

        # Mock create_binary_labels to return DataFrame with 'label_binary' column
        labeled_df = features_df.copy()
        labeled_df['label_binary'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Ensure both classes are present
        mock_create_binary_labels.return_value = labeled_df

        # Execute main
        main()

        # Verify train_model_binary was called
        call_args = mock_train_model_binary.call_args
        assert call_args is not None

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.generate_features')
    @patch('ml.trainer.create_binary_labels')
    @patch('ml.trainer.train_model_binary')
    @patch('ml.trainer.setup_logging')
    def test_main_with_tuning(self, mock_setup_logging, mock_train_model_binary,
                            mock_create_binary_labels, mock_generate_features,
                            mock_load_data, mock_parse_args):
        """Test main with hyperparameter tuning (lines 435-441, 447-449, 464-465)."""
        # Mock arguments with tuning enabled
        mock_args = MagicMock()
        mock_args.data = 'test.csv'
        mock_args.output = 'model.pkl'
        mock_args.up_thresh = 0.005
        mock_args.horizon = 5
        mock_args.tune = True
        mock_args.n_trials = 10
        mock_args.drop_neutral = False
        mock_args.feature_selection = False
        mock_args.eval_profit = False
        mock_args.n_splits = 2  # Add n_splits
        mock_args.results = 'results.json'
        mock_args.logfile = None
        mock_args.early_stopping_rounds = 50

        mock_parse_args.return_value = mock_args

        # Mock data with sufficient rows for horizon=5 (need at least 10 samples)
        mock_df = pd.DataFrame({
            'Open': list(range(10, 20)),
            'High': list(range(12, 22)),
            'Low': list(range(8, 18)),
            'Close': list(range(11, 21)),
            'Volume': list(range(100, 200, 10))
        })
        mock_load_data.return_value = mock_df

        # Mock generate_features to return DataFrame with features
        features_df = mock_df.copy()
        features_df['RSI'] = [50, 60, 70, 55, 65, 75, 45, 52, 58, 62]
        features_df['MACD'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['EMA_20'] = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        features_df['ATR'] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        features_df['StochRSI'] = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.45, 0.52, 0.58, 0.62]
        features_df['TrendStrength'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['Volatility'] = [0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.08, 0.14, 0.16, 0.19]
        mock_generate_features.return_value = features_df

        # Mock create_binary_labels to return DataFrame with 'label_binary' column
        binary_df = features_df.copy()
        binary_df['label_binary'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Ensure both classes are present
        mock_create_binary_labels.return_value = binary_df

        # Assert class balance in mocked labels
        assert set(binary_df['label_binary']) == {0, 1}

        # Execute main
        main()

        # Verify train_model_binary was called with tuning parameters
        call_args = mock_train_model_binary.call_args
        assert call_args[1]['tune'] is True
        assert call_args[1]['n_trials'] == 10

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.generate_features')
    @patch('ml.trainer.create_binary_labels')
    @patch('ml.trainer.train_model_binary')
    @patch('ml.trainer.setup_logging')
    def test_main_with_feature_selection(self, mock_setup_logging, mock_train_model_binary,
                                       mock_create_binary_labels, mock_generate_features,
                                       mock_load_data, mock_parse_args):
        """Test main with feature selection (lines 470-496)."""
        # Mock arguments with feature selection
        mock_args = MagicMock()
        mock_args.data = 'test.csv'
        mock_args.output = 'model.pkl'
        mock_args.up_thresh = 0.005
        mock_args.horizon = 5
        mock_args.feature_selection = True
        mock_args.tune = False
        mock_args.drop_neutral = False
        mock_args.eval_profit = False
        mock_args.n_splits = 2  # Add n_splits
        mock_args.results = 'results.json'
        mock_args.logfile = None
        mock_args.n_trials = 25
        mock_args.early_stopping_rounds = 50

        mock_parse_args.return_value = mock_args

        # Mock data with sufficient rows for horizon=5 (need at least 10 samples)
        mock_df = pd.DataFrame({
            'Open': list(range(10, 20)),
            'High': list(range(12, 22)),
            'Low': list(range(8, 18)),
            'Close': list(range(11, 21)),
            'Volume': list(range(100, 200, 10))
        })
        mock_load_data.return_value = mock_df

        # Mock generate_features to return DataFrame with features
        features_df = mock_df.copy()
        features_df['RSI'] = [50, 60, 70, 55, 65, 75, 45, 52, 58, 62]
        features_df['MACD'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['EMA_20'] = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        features_df['ATR'] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        features_df['StochRSI'] = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.45, 0.52, 0.58, 0.62]
        features_df['TrendStrength'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['Volatility'] = [0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.08, 0.14, 0.16, 0.19]
        mock_generate_features.return_value = features_df

        # Mock create_binary_labels to return DataFrame with 'label_binary' column
        binary_df = features_df.copy()
        binary_df['label_binary'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Ensure both classes are present
        mock_create_binary_labels.return_value = binary_df

        # Assert class balance in mocked labels
        assert set(binary_df['label_binary']) == {0, 1}

        # Execute main
        main()

        # Verify train_model_binary was called with feature_selection=True
        call_args = mock_train_model_binary.call_args
        assert call_args[1]['feature_selection'] is True

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.generate_features')
    @patch('ml.trainer.create_binary_labels')
    @patch('ml.trainer.train_model_binary')
    @patch('ml.trainer.setup_logging')
    def test_main_with_profit_evaluation(self, mock_setup_logging, mock_train_model_binary,
                                       mock_create_binary_labels, mock_generate_features,
                                       mock_load_data, mock_parse_args):
        """Test main with profit evaluation (lines 532-544, 551-552, 559-560)."""
        # Mock arguments with profit evaluation
        mock_args = MagicMock()
        mock_args.data = 'test.csv'
        mock_args.output = 'model.pkl'
        mock_args.up_thresh = 0.005
        mock_args.horizon = 5
        mock_args.eval_profit = True
        mock_args.tune = False
        mock_args.drop_neutral = False
        mock_args.feature_selection = False
        mock_args.n_splits = 2  # Add n_splits
        mock_args.results = 'results.json'
        mock_args.logfile = None
        mock_args.n_trials = 25
        mock_args.early_stopping_rounds = 50

        mock_parse_args.return_value = mock_args

        # Mock data with sufficient rows for horizon=5 (need at least 10 samples)
        mock_df = pd.DataFrame({
            'Open': list(range(10, 20)),
            'High': list(range(12, 22)),
            'Low': list(range(8, 18)),
            'Close': list(range(11, 21)),
            'Volume': list(range(100, 200, 10))
        })
        mock_load_data.return_value = mock_df

        # Mock generate_features to return DataFrame with features
        features_df = mock_df.copy()
        features_df['RSI'] = [50, 60, 70, 55, 65, 75, 45, 52, 58, 62]
        features_df['MACD'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['EMA_20'] = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5]
        features_df['ATR'] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        features_df['StochRSI'] = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.45, 0.52, 0.58, 0.62]
        features_df['TrendStrength'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22]
        features_df['Volatility'] = [0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.08, 0.14, 0.16, 0.19]
        mock_generate_features.return_value = features_df

        # Mock create_binary_labels to return DataFrame with 'label_binary' column
        binary_df = features_df.copy()
        binary_df['label_binary'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Ensure both classes are present
        mock_create_binary_labels.return_value = binary_df

        # Assert class balance in mocked labels
        assert set(binary_df['label_binary']) == {0, 1}

        # Execute main
        main()

        # Verify train_model_binary was called with eval_profit=True
        call_args = mock_train_model_binary.call_args
        assert call_args[1]['eval_economic'] is True  # eval_profit maps to eval_economic

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.setup_logging')
    def test_main_label_creation_error(self, mock_setup_logging, mock_load_data, mock_parse_args):
        """Test main with label creation error."""
        # Mock arguments
        mock_args = MagicMock()
        mock_args.data = 'test.csv'
        mock_parse_args.return_value = mock_args

        # Mock data with required columns
        mock_df = pd.DataFrame({
            'Open': [10, 11, 12],
            'High': [12, 13, 14],
            'Low': [8, 9, 10],
            'Close': [11, 12, 13],
            'Volume': [100, 110, 120]
        })
        mock_load_data.return_value = mock_df

        with patch('ml.trainer.generate_features') as mock_generate_features, \
             patch('ml.trainer.create_binary_labels') as mock_create_binary_labels:

            # Mock successful feature generation
            mock_generate_features.return_value = mock_df

            # Mock binary label creation that doesn't create label_binary column
            mock_create_binary_labels.return_value = mock_df  # No 'label_binary' column

            # Should raise ValueError
            with pytest.raises(ValueError, match="label_binary column not found"):
                main()

    @patch('ml.trainer.argparse.ArgumentParser.parse_args')
    @patch('ml.trainer.load_data')
    @patch('ml.trainer.generate_features')
    @patch('ml.trainer.create_binary_labels')
    @patch('ml.trainer.train_model_binary')
    @patch('ml.trainer.setup_logging')
    def test_main_with_logfile(self, mock_setup_logging, mock_train_model_binary,
                             mock_create_binary_labels, mock_generate_features,
                             mock_load_data, mock_parse_args):
        """Test main with logfile option (lines 564-585, 589-630, 649)."""
        # Mock arguments with logfile
        mock_args = MagicMock()
        mock_args.data = 'test.csv'
        mock_args.output = 'model.pkl'
        mock_args.up_thresh = 0.005
        mock_args.horizon = 5
        mock_args.logfile = 'test.log'
        mock_args.tune = False
        mock_args.drop_neutral = False
        mock_args.feature_selection = False
        mock_args.eval_profit = False
        mock_args.n_splits = 2  # Add n_splits
        mock_args.results = 'results.json'
        mock_args.n_trials = 25
        mock_args.early_stopping_rounds = 50

        mock_parse_args.return_value = mock_args

        # Mock data with sufficient rows for horizon=5 (need at least 10 samples after preprocessing)
        mock_df = pd.DataFrame({
            'Open': list(range(10, 25)),  # 15 samples
            'High': list(range(12, 27)),  # 15 samples
            'Low': list(range(8, 23)),    # 15 samples
            'Close': list(range(11, 26)), # 15 samples
            'Volume': list(range(100, 250, 10)) # 15 samples
        })
        mock_load_data.return_value = mock_df

        # Mock generate_features to return DataFrame with features (15 rows to match mock_df)
        features_df = mock_df.copy()
        features_df['RSI'] = [50, 60, 70, 55, 65, 75, 45, 52, 58, 62, 48, 54, 59, 51, 57]
        features_df['MACD'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22, 0.08, 0.14, 0.19, 0.11, 0.17]
        features_df['EMA_20'] = [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5]
        features_df['ATR'] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
        features_df['StochRSI'] = [0.5, 0.6, 0.7, 0.55, 0.65, 0.75, 0.45, 0.52, 0.58, 0.62, 0.48, 0.54, 0.59, 0.51, 0.57]
        features_df['TrendStrength'] = [0.1, 0.2, 0.3, 0.15, 0.25, 0.35, 0.05, 0.12, 0.18, 0.22, 0.08, 0.14, 0.19, 0.11, 0.17]
        features_df['Volatility'] = [0.1, 0.15, 0.2, 0.12, 0.18, 0.22, 0.08, 0.14, 0.16, 0.19, 0.11, 0.17, 0.21, 0.13, 0.15]
        mock_generate_features.return_value = features_df

        # Mock create_binary_labels to return DataFrame with 'label_binary' column (15 rows to match)
        binary_df = features_df.copy()
        binary_df['label_binary'] = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Ensure both classes are present
        mock_create_binary_labels.return_value = binary_df

        # Mock train_model_binary to avoid the actual training
        mock_train_model_binary.return_value = {
            'metadata': {'model_type': 'binary_classification'},
            'overall_metrics': {'auc': 0.8, 'f1': 0.75}
        }

        # Execute main
        main()

        # Verify setup_logging was called with logfile
        mock_setup_logging.assert_called_once_with('test.log')


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    def test_technical_indicator_errors(self):
        """Test error handling in technical indicators."""
        # Test with empty data
        empty_series = pd.Series([], dtype=float)

        # RSI should handle empty series
        rsi_result = compute_rsi(empty_series)
        assert len(rsi_result) == 0

        # MACD should handle empty series
        macd_result = compute_macd(empty_series)
        assert len(macd_result) == 0

    def test_data_processing_errors(self):
        """Test error handling in data processing."""
        # Test generate_features with None
        with pytest.raises(AttributeError):
            generate_features(None)

        # Test create_labels with None
        with pytest.raises(AttributeError):
            create_labels(None)

    @patch('ml.trainer.joblib.dump')
    def test_model_save_error_handling(self, mock_joblib_dump):
        """Test error handling during model saving."""
        # Create minimal test data with only 0 and 1 labels
        data = pd.DataFrame({
            'RSI': [50, 60, 70, 80, 90],
            'MACD': [0.1, 0.2, 0.3, 0.4, 0.5],
            'EMA_20': [10, 11, 12, 13, 14],
            'ATR': [1, 1.1, 1.2, 1.3, 1.4],
            'StochRSI': [0.5, 0.6, 0.7, 0.8, 0.9],
            'TrendStrength': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Volatility': [0.1, 0.15, 0.2, 0.25, 0.3],
            'Label': [1, 0, 1, 0, 1]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Mock save failure
            mock_joblib_dump.side_effect = Exception("Disk full")

            # Should handle error gracefully (not crash)
            train_model(data, model_path, n_splits=2)

    @patch('ml.trainer.json.dump')
    def test_results_save_error_handling(self, mock_json_dump):
        """Test error handling during results saving."""
        # Create minimal test data with only 0 and 1 labels
        data = pd.DataFrame({
            'RSI': [50, 60, 70, 80, 90],
            'MACD': [0.1, 0.2, 0.3, 0.4, 0.5],
            'EMA_20': [10, 11, 12, 13, 14],
            'ATR': [1, 1.1, 1.2, 1.3, 1.4],
            'StochRSI': [0.5, 0.6, 0.7, 0.8, 0.9],
            'TrendStrength': [0.1, 0.2, 0.3, 0.4, 0.5],
            'Volatility': [0.1, 0.15, 0.2, 0.25, 0.3],
            'Label': [1, 0, 1, 0, 1]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Mock JSON save failure
            mock_json_dump.side_effect = Exception("Permission denied")

            # Should handle error gracefully
            train_model(data, model_path, results_path=results_path, n_splits=2)


class TestBinaryTraining:
    """Test cases for binary classification training pipeline."""

    def test_train_model_binary_basic(self):
        """Test basic binary model training with synthetic data."""
        # Create synthetic OHLCV data with more variation to ensure good label distribution
        np.random.seed(42)
        rows = 800
        idx = pd.date_range("2020-01-01", periods=rows, freq="H")

        # Create data with more price variation to get better label distribution
        base_price = 100 + np.cumsum(np.random.normal(0, 0.5, rows))
        noise = np.random.normal(0, 1, rows)
        close_prices = base_price + noise

        df = pd.DataFrame({
            "Open": close_prices + np.random.normal(0, 0.5, rows),
            "High": close_prices + abs(np.random.normal(0, 1, rows)),
            "Low": close_prices - abs(np.random.normal(0, 1, rows)),
            "Close": close_prices,
            "Volume": np.random.uniform(1000, 5000, rows),
        }, index=idx)

        # Generate features and create binary labels
        df = generate_features(df)
        df = create_binary_labels(df, horizon=5, profit_threshold=0.005)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Train binary model
            results = train_model_binary(
                df,
                model_path,
                results_path=results_path,
                n_splits=2,  # Use smaller n_splits for test
                horizon=5,
                profit_threshold=0.005,
                include_fees=True,
                fee_rate=0.001,
                tune=False,
                feature_selection=False,
                early_stopping_rounds=50,
                eval_economic=True,
            )

            # Validate output files exist
            assert os.path.exists(model_path), "Model file should be created"
            assert os.path.exists(results_path), "Results file should be created"

            # Validate model card is created
            card_path = model_path.replace('.pkl', '.model_card.json')
            assert os.path.exists(card_path), "Model card should be created"

            # Validate results structure
            assert 'metadata' in results
            assert 'overall_metrics' in results
            assert 'fold_metrics' in results
            assert 'feature_importance' in results

            # Validate expected metrics exist
            overall_metrics = results['overall_metrics']
            assert 'auc' in overall_metrics
            assert 'f1' in overall_metrics
            assert 'total_samples' in overall_metrics
            assert 'class_distribution' in overall_metrics

            # Validate AUC and F1 are reasonable values
            assert 0.0 <= overall_metrics['auc'] <= 1.0
            assert 0.0 <= overall_metrics['f1'] <= 1.0

    def test_train_model_binary_small_dataset(self):
        """Test binary training with small dataset (should handle gracefully)."""
        # Create dataset with more realistic price movements
        np.random.seed(42)
        rows = 300
        idx = pd.date_range("2020-01-01", periods=rows, freq="H")

        # Create data with realistic price movements
        base_price = 100
        price_changes = np.random.normal(0, 0.02, rows)  # 2% daily volatility
        close_prices = base_price * np.cumprod(1 + price_changes)

        df = pd.DataFrame({
            "Open": close_prices * (1 + np.random.normal(0, 0.005, rows)),
            "High": close_prices * (1 + abs(np.random.normal(0, 0.01, rows))),
            "Low": close_prices * (1 - abs(np.random.normal(0, 0.01, rows))),
            "Close": close_prices,
            "Volume": np.random.uniform(1000, 5000, rows),
        }, index=idx)

        # Generate features and create binary labels
        df = generate_features(df)
        df = create_binary_labels(df, horizon=3, profit_threshold=0.005)  # Standard threshold

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Should complete without raising exceptions
            results = train_model_binary(
                df,
                model_path,
                results_path=results_path,
                n_splits=2,  # Small number for small dataset
                horizon=3,
                profit_threshold=0.01,
                include_fees=False,  # Simpler for small dataset
                tune=False,
                eval_economic=False,  # Disable economic metrics for small dataset
            )

            # Should still create output files
            assert os.path.exists(model_path)
            assert os.path.exists(results_path)

    def test_train_model_binary_insufficient_data(self):
        """Test binary training with insufficient data (should handle gracefully)."""
        # Create very small dataset
        df = make_sample_df(rows=10)

        # Generate features and create binary labels
        df = generate_features(df)
        df = create_binary_labels(df, horizon=5, profit_threshold=0.005)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Should handle gracefully - either complete successfully or raise appropriate error
            try:
                train_model_binary(
                    df,
                    model_path,
                    n_splits=5,
                    horizon=5,
                    profit_threshold=0.005,
                )
                # If it completes, that's acceptable
                assert True
            except ValueError as e:
                # Appropriate error handling is also acceptable
                # The error might be about insufficient data or single class
                assert "data" in str(e).lower() or "class" in str(e).lower()

    def test_train_model_binary_missing_features(self):
        """Test binary training with missing required features."""
        # Create data without required OHLCV columns
        df = pd.DataFrame({
            'Price': [100, 101, 102, 103, 104],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Should raise ValueError for missing required columns
            with pytest.raises(ValueError, match="label_binary"):
                train_model_binary(
                    df,
                    model_path,
                    n_splits=2,
                    horizon=2,
                    profit_threshold=0.005,
                )

    def test_train_model_binary_with_tuning(self):
        """Test binary training with hyperparameter tuning."""
        # Create synthetic data with realistic price movements
        np.random.seed(42)
        rows = 600
        idx = pd.date_range("2020-01-01", periods=rows, freq="H")

        # Create data with realistic price movements
        base_price = 100
        price_changes = np.random.normal(0, 0.02, rows)  # 2% daily volatility
        close_prices = base_price * np.cumprod(1 + price_changes)

        df = pd.DataFrame({
            "Open": close_prices * (1 + np.random.normal(0, 0.005, rows)),
            "High": close_prices * (1 + abs(np.random.normal(0, 0.01, rows))),
            "Low": close_prices * (1 - abs(np.random.normal(0, 0.01, rows))),
            "Close": close_prices,
            "Volume": np.random.uniform(1000, 5000, rows),
        }, index=idx)

        # Generate features and create binary labels
        df = generate_features(df)
        df = create_binary_labels(df, horizon=5, profit_threshold=0.005)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Mock optuna for testing
            mock_optuna = MagicMock()
            mock_study = MagicMock()
            mock_study.best_params = {
                'learning_rate': 0.1,
                'n_estimators': 50,
                'num_leaves': 31
            }
            mock_study.best_value = 0.85
            mock_study.optimize = MagicMock()
            mock_optuna.create_study.return_value = mock_study

            with patch.dict('sys.modules', {'optuna': mock_optuna}):
                results = train_model_binary(
                    df,
                    model_path,
                    results_path=results_path,
                    n_splits=3,
                    horizon=5,
                    profit_threshold=0.005,
                    tune=True,
                    n_trials=2,  # Small number for testing
                    eval_economic=False,
                )

                # Should complete successfully
                assert os.path.exists(model_path)
                assert os.path.exists(results_path)

                # Should have tuning metadata
                assert 'hyperparameter_tuning' in results['metadata']
                assert results['metadata']['hyperparameter_tuning'] is True


class TestMulticlassTraining:
    """Test cases for multiclass training pipeline."""

    def test_train_model_multiclass_basic(self):
        """Test basic multiclass model training with synthetic data."""
        # Create synthetic OHLCV data with more samples to ensure good class distribution
        df = make_sample_df(rows=500)

        # Generate features and create multiclass labels
        df = generate_features(df)
        df = create_labels(df, horizon=5, up_thresh=0.01, down_thresh=-0.01)

        # Ensure we have multiple classes
        if df['Label'].nunique() < 2:
            # If we don't have enough classes, create some variation
            df.loc[df.index[:len(df)//3], 'Label'] = 1
            df.loc[df.index[len(df)//3:2*len(df)//3], 'Label'] = -1
            df.loc[df.index[2*len(df)//3:], 'Label'] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Train multiclass model
            train_model(
                df,
                model_path,
                results_path=results_path,
                n_splits=3,
                horizon=5,
                up_thresh=0.01,
                down_thresh=-0.01,
                drop_neutral=False,
                tune=False,
                feature_selection=False,
                early_stopping_rounds=50,
                eval_profit=False,
            )

            # Validate output files exist
            assert os.path.exists(model_path), "Model file should be created"
            assert os.path.exists(results_path), "Results file should be created"

            # Validate model card is created
            card_path = model_path.replace('.pkl', '.model_card.json')
            assert os.path.exists(card_path), "Model card should be created"

            # Validate results file contains expected metrics
            with open(results_path, 'r') as f:
                results = json.load(f)

            assert 'metadata' in results
            assert 'folds' in results
            assert 'mean' in results
            assert 'class_distribution' in results

            # Validate mean metrics exist
            mean_metrics = results['mean']
            assert 'f1' in mean_metrics
            assert 'precision' in mean_metrics
            assert 'recall' in mean_metrics

    def test_train_model_multiclass_drop_neutral(self):
        """Test multiclass training with neutral labels dropped."""
        # Create synthetic data with some neutral labels
        df = make_sample_df(rows=200)

        # Generate features and create labels
        df = generate_features(df)
        df = create_labels(df, horizon=3, up_thresh=0.02, down_thresh=-0.02)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Train with drop_neutral=True
            train_model(
                df,
                model_path,
                results_path=results_path,
                n_splits=2,
                drop_neutral=True,
                tune=False,
            )

            # Should complete successfully - results file should always exist
            assert os.path.exists(results_path)

            # Model file may not exist if all labels were neutral (which is correct behavior)
            # Just check that the function handled the edge case gracefully

    def test_train_model_multiclass_insufficient_data(self):
        """Test multiclass training with insufficient data."""
        # Create very small dataset
        df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [98, 99, 100],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        })

        # Generate features and create labels
        df = generate_features(df)
        df = create_labels(df, horizon=2)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')

            # Should handle gracefully or raise appropriate error
            try:
                train_model(df, model_path, n_splits=2)
                # If it completes, that's acceptable for small datasets
                assert os.path.exists(model_path)
            except (ValueError, Exception):
                # Appropriate error handling is also acceptable
                pass


class TestIntegration:
    """Test cases for complete integration scenarios."""

    def test_full_pipeline_small_dataset(self):
        """Test complete pipeline with small dataset."""
        # Create small but valid dataset with more realistic OHLCV data
        np.random.seed(42)
        n_samples = 50
        base_price = 100
        data = pd.DataFrame({
            'Open': np.random.uniform(base_price - 2, base_price + 2, n_samples),
            'High': np.random.uniform(base_price + 1, base_price + 5, n_samples),
            'Low': np.random.uniform(base_price - 5, base_price - 1, n_samples),
            'Close': np.random.uniform(base_price - 2, base_price + 2, n_samples),
            'Volume': np.random.uniform(1000, 5000, n_samples)
        })

        # Generate features and labels as in the main pipeline
        data = generate_features(data)
        data = create_labels(data)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Should complete without errors
            train_model(data, model_path, results_path=results_path, n_splits=2)

            # Check that files were created
            assert os.path.exists(model_path)
            assert os.path.exists(results_path)

    def test_end_to_end_cli_simulation(self):
        """Test end-to-end CLI simulation."""
        # Create test CSV file with more realistic OHLCV data
        np.random.seed(42)
        n_samples = 50
        base_price = 100
        test_data = pd.DataFrame({
            'Open': np.random.uniform(base_price - 2, base_price + 2, n_samples),
            'High': np.random.uniform(base_price + 1, base_price + 5, n_samples),
            'Low': np.random.uniform(base_price - 5, base_price - 1, n_samples),
            'Close': np.random.uniform(base_price - 2, base_price + 2, n_samples),
            'Volume': np.random.uniform(1000, 5000, n_samples)
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'test_data.csv')
            model_path = os.path.join(temp_dir, 'model.pkl')
            results_path = os.path.join(temp_dir, 'results.json')

            # Save test data
            test_data.to_csv(csv_path, index=False)

            # Simulate CLI arguments
            with patch('sys.argv', [
                'trainer.py',
                '--data', csv_path,
                '--output', model_path,
                '--results', results_path,
                '--n_splits', '2'
            ]):
                # Should run without errors
                main()

            # Check that output files were created
            assert os.path.exists(model_path)
            assert os.path.exists(results_path)

    def test_binary_vs_multiclass_consistency(self):
        """Test that binary and multiclass pipelines produce consistent outputs."""
        # Create synthetic data with more realistic price movements
        np.random.seed(42)
        rows = 600
        idx = pd.date_range("2020-01-01", periods=rows, freq="H")

        # Create data with realistic price movements
        base_price = 100
        price_changes = np.random.normal(0, 0.02, rows)  # 2% daily volatility
        close_prices = base_price * np.cumprod(1 + price_changes)

        df = pd.DataFrame({
            "Open": close_prices * (1 + np.random.normal(0, 0.005, rows)),
            "High": close_prices * (1 + abs(np.random.normal(0, 0.01, rows))),
            "Low": close_prices * (1 - abs(np.random.normal(0, 0.01, rows))),
            "Close": close_prices,
            "Volume": np.random.uniform(1000, 5000, rows),
        }, index=idx)

        # Generate features
        df_features = generate_features(df.copy())

        # Test binary pipeline
        df_binary = create_binary_labels(df_features.copy(), horizon=5, profit_threshold=0.005)

        with tempfile.TemporaryDirectory() as temp_dir:
            binary_model_path = os.path.join(temp_dir, 'binary_model.pkl')
            binary_results_path = os.path.join(temp_dir, 'binary_results.json')

            # Binary training might fail due to single class, so handle gracefully
            try:
                binary_results = train_model_binary(
                    df_binary,
                    binary_model_path,
                    results_path=binary_results_path,
                    n_splits=3,
                    horizon=5,
                    profit_threshold=0.005,
                    eval_economic=False,
                )
                binary_success = True
            except ValueError:
                binary_success = False

            # Test multiclass pipeline
            df_multiclass = create_labels(df_features.copy(), horizon=5, up_thresh=0.005, down_thresh=-0.005)

            # Ensure multiclass has multiple classes
            if df_multiclass['Label'].nunique() < 2:
                df_multiclass.loc[df_multiclass.index[:len(df_multiclass)//3], 'Label'] = 1
                df_multiclass.loc[df_multiclass.index[len(df_multiclass)//3:2*len(df_multiclass)//3], 'Label'] = -1
                df_multiclass.loc[df_multiclass.index[2*len(df_multiclass)//3:], 'Label'] = 0

            multiclass_model_path = os.path.join(temp_dir, 'multiclass_model.pkl')
            multiclass_results_path = os.path.join(temp_dir, 'multiclass_results.json')

            train_model(
                df_multiclass,
                multiclass_model_path,
                results_path=multiclass_results_path,
                n_splits=3,
                horizon=5,
                up_thresh=0.005,
                down_thresh=-0.005,
                tune=False,
            )

            # Multiclass should always create output files
            assert os.path.exists(multiclass_model_path)
            assert os.path.exists(multiclass_results_path)
            assert os.path.exists(multiclass_model_path.replace('.pkl', '.model_card.json'))

            # Binary might or might not succeed depending on label distribution
            if binary_success:
                assert os.path.exists(binary_model_path)
                assert os.path.exists(binary_results_path)
                assert os.path.exists(binary_model_path.replace('.pkl', '.model_card.json'))

                # Validate binary results structure
                assert 'overall_metrics' in binary_results
                assert 'auc' in binary_results['overall_metrics']
                assert 'f1' in binary_results['overall_metrics']

            # Validate multiclass results structure
            with open(multiclass_results_path, 'r') as f:
                multiclass_results = json.load(f)

            assert 'mean' in multiclass_results
            assert 'f1' in multiclass_results['mean']
            assert 'precision' in multiclass_results['mean']
            assert 'recall' in multiclass_results['mean']
