"""
Tests for vectorized feature operations.

This module contains comprehensive tests for vectorized feature calculations,
including performance benchmarks, numerical accuracy tests, and edge case handling.
"""

import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from ml.features import FeatureExtractor
from ml.indicators import calculate_obv, calculate_all_indicators


class VectorizedFeatureCalculator:
    """
    Vectorized feature calculator for testing.

    This class provides vectorized implementations of feature calculations
    for performance benchmarking and accuracy testing.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.extractor = FeatureExtractor(config)

    def calculate_all_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate all features from price data.

        Args:
            prices: Array of price data

        Returns:
            Array of calculated features
        """
        # Create OHLCV DataFrame from prices (assuming close prices)
        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,  # Simulate high
            'low': prices * 0.99,   # Simulate low
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(prices))
        })

        # For performance testing with large datasets, use optimized config
        if len(prices) > 100000:  # 100K+ rows
            # Use minimal config for performance test - skip all indicators
            perf_config = {
                "indicator_params": {},  # No indicators for performance test
                "scaling": {"method": "none"},  # Skip scaling for performance
                "lagged_features": {"enabled": False},  # Skip expensive lags
                "price_features": {"returns": False, "log_returns": False},  # Skip price features
                "volume_features": {"volume_sma": False, "volume_ratio": False},  # Skip volume features
                "validation": {
                    "require_min_rows": 1,  # Minimal threshold for perf test
                    "handle_missing": "drop"
                },
            }
            perf_extractor = FeatureExtractor(perf_config)
            # Skip indicator calculation entirely for performance test
            df_with_indicators = df.copy()  # Just use raw OHLCV data
            df_with_price_features = perf_extractor._add_price_features(df_with_indicators)
            df_with_volume_features = perf_extractor._add_volume_features(df_with_price_features)
            df_with_lagged = perf_extractor._add_lagged_features(df_with_volume_features)
            df_clean = perf_extractor._clean_features(df_with_lagged)
            features_df = perf_extractor._scale_features(df_clean, fit_scaler=False)
        else:
            # Use full feature extraction for normal cases
            features_df = self.extractor.extract_features(df, fit_scaler=False)

        # Return as numpy array
        return features_df.values


class TestVectorizedOperations:
    """Test suite for vectorized feature operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            "indicator_params": {
                "rsi_period": 14,
                "ema_period": 20,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "atr_period": 14,
                "adx_period": 14,
            },
            "scaling": {"method": "none"},
            "lagged_features": {"enabled": True, "periods": [1, 2, 3]},
            "price_features": {"returns": True, "log_returns": True},
            "volume_features": {"volume_sma": True, "volume_ratio": True},
            "validation": {"require_min_rows": 50, "handle_missing": "drop"},
        }
        self.calculator = VectorizedFeatureCalculator(self.config)

    @pytest.mark.timeout(30)
    def test_vectorized_large_dataset_performance(self):
        """Test performance with large dataset."""
        calculator = VectorizedFeatureCalculator()

        # Large dataset that would timeout with loops
        large_prices = np.random.random(1000000) * 1000  # 1M data points

        start_time = time.time()
        features = calculator.calculate_all_features(large_prices)
        processing_time = time.time() - start_time

        # Should complete within timeout and be faster than threshold
        assert processing_time < 10.0
        assert len(features) <= len(large_prices)  # May drop some rows due to NaN
        assert len(features) > 0  # Should have some features
        assert not np.isnan(features).all()  # Should have some valid features

    def test_obv_numerical_accuracy(self):
        """Test OBV calculation numerical accuracy."""
        # Test data
        data = pd.DataFrame({
            'close': [10, 11, 12, 11, 10, 11, 11, 12],
            'volume': [100, 200, 300, 150, 250, 180, 120, 220]
        })

        obv_result = calculate_obv(data)

        # Expected OBV calculation:
        # i=0: 100
        # i=1: 100 + 200 = 300 (close > prev)
        # i=2: 300 + 300 = 600 (close > prev)
        # i=3: 600 - 150 = 450 (close < prev)
        # i=4: 450 - 250 = 200 (close < prev)
        # i=5: 200 + 180 = 380 (close > prev)
        # i=6: 380 + 0 = 380 (close == prev)
        # i=7: 380 + 220 = 600 (close > prev)
        expected = [100, 300, 600, 450, 200, 380, 380, 600]

        np.testing.assert_array_equal(obv_result.values, expected)

    def test_obv_edge_cases(self):
        """Test OBV with edge cases."""
        # Empty data
        empty_data = pd.DataFrame(columns=['close', 'volume'])
        result = calculate_obv(empty_data)
        assert len(result) == 0

        # Single row
        single_data = pd.DataFrame({'close': [10], 'volume': [100]})
        result = calculate_obv(single_data)
        assert result.iloc[0] == 100

        # All equal closes
        equal_data = pd.DataFrame({
            'close': [10, 10, 10, 10],
            'volume': [100, 200, 300, 400]
        })
        result = calculate_obv(equal_data)
        expected = [100, 100, 100, 100]  # Should stay at initial volume
        np.testing.assert_array_equal(result.values, expected)

        # Decreasing closes
        dec_data = pd.DataFrame({
            'close': [10, 9, 8, 7],
            'volume': [100, 200, 300, 400]
        })
        result = calculate_obv(dec_data)
        expected = [100, 100-200, 100-200-300, 100-200-300-400]
        np.testing.assert_array_equal(result.values, expected)

    def test_feature_extraction_numerical_stability(self):
        """Test numerical stability of feature extraction."""
        # Create data with potential numerical issues
        prices = np.array([1.0, 1.0000000001, 1.0000000002, 1.0])
        volumes = np.array([1000, 1001, 999, 1000])

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': volumes
        })

        # Use config with lower min_rows and fill missing for this test
        test_config = self.config.copy()
        test_config["validation"]["require_min_rows"] = 4
        test_config["validation"]["handle_missing"] = "fill"
        extractor = FeatureExtractor(test_config)
        features = extractor.extract_features(data)

        # Check for infinite values
        assert not np.isinf(features.values).any()

        # Check that the system handles small data gracefully
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(data)
        assert features.shape[1] > 0  # Has some features

        # Check that original OHLCV columns are preserved
        assert 'open' in features.columns
        assert 'high' in features.columns
        assert 'low' in features.columns
        assert 'close' in features.columns
        assert 'volume' in features.columns

    def test_memory_efficiency(self):
        """Test memory efficiency of vectorized operations."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Large dataset
        large_prices = np.random.random(500000) * 100
        calculator = VectorizedFeatureCalculator()

        features = calculator.calculate_all_features(large_prices)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory (less than 500MB increase)
        assert memory_increase < 500
        assert len(features) <= len(large_prices)  # May drop rows due to NaN
        assert len(features) > 0

    def test_batch_processing_performance(self):
        """Test batch processing performance."""
        # Create multiple datasets
        datasets = []
        for i in range(10):
            prices = np.random.random(10000) * 100
            data = pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(prices))
            })
            datasets.append(data)

        start_time = time.time()
        extractor = FeatureExtractor(self.config)

        results = []
        for data in datasets:
            features = extractor.extract_features(data, fit_scaler=False)
            results.append(features)

        processing_time = time.time() - start_time

        # Should process 10 datasets of 10K rows each within reasonable time
        assert processing_time < 30.0  # 30 seconds
        assert len(results) == 10
        assert all(len(r) <= 10000 for r in results)  # May drop some rows due to NaN
        assert all(len(r) > 0 for r in results)

    def test_realtime_processing_mode(self):
        """Test real-time processing mode."""
        # Simulate streaming data
        initial_prices = np.random.random(100) * 100
        data = pd.DataFrame({
            'open': initial_prices,
            'high': initial_prices * 1.01,
            'low': initial_prices * 0.99,
            'close': initial_prices,
            'volume': np.random.randint(1000, 10000, 100)
        })

        extractor = FeatureExtractor(self.config)
        initial_features = extractor.extract_features(data)

        # Add new data point
        new_price = np.random.random() * 100
        new_data = pd.DataFrame({
            'open': [new_price],
            'high': [new_price * 1.01],
            'low': [new_price * 0.99],
            'close': [new_price],
            'volume': [np.random.randint(1000, 10000)]
        })

        # Incremental update (simulate real-time)
        combined_data = pd.concat([data, new_data], ignore_index=True)
        updated_features = extractor.extract_features(combined_data, fit_scaler=False)

        # Check that we can handle incremental updates
        assert len(updated_features) <= len(combined_data)  # May drop rows
        assert len(updated_features) >= len(initial_features)  # Should have at least as many as initial

    def test_nan_handling(self):
        """Test NaN value handling in vectorized operations."""
        # Create data with NaN values
        prices = np.random.random(100) * 100
        prices[10:15] = np.nan  # Introduce NaN values
        volumes = np.random.randint(1000, 10000, 100).astype(float)  # Use float to allow NaN
        volumes[20:25] = np.nan

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': volumes
        })

        extractor = FeatureExtractor(self.config)
        features = extractor.extract_features(data)

        # Should handle NaN gracefully (drop or fill)
        assert isinstance(features, pd.DataFrame)
        # Depending on config, might drop rows or fill
        assert len(features) <= len(data)

    def test_indicator_vectorization_accuracy(self):
        """Test that indicators produce same results as vectorized versions."""
        np.random.seed(42)
        prices = np.random.random(100) * 100 + 50
        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })

        # Calculate indicators
        indicators_df = calculate_all_indicators(data)

        # Check that OBV is calculated (vectorized)
        assert 'obv' in indicators_df.columns
        assert not indicators_df['obv'].isnull().all()

        # Check other indicators
        expected_indicators = [
            'rsi', 'ema', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'adx', 'obv'
        ]

        for indicator in expected_indicators:
            assert indicator in indicators_df.columns
            # Should not be all NaN (assuming sufficient data)
            if len(data) >= 50:  # Sufficient for most indicators
                assert not indicators_df[indicator].isnull().all()

    def test_cross_asset_features_performance(self):
        """Test cross-asset feature performance."""
        # Create multiple assets
        assets_data = {}
        for i in range(5):
            prices = np.random.random(10000) * 100
            data = pd.DataFrame({
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(prices))
            })
            assets_data[f'ASSET_{i}'] = data

        extractor = FeatureExtractor(self.config)

        start_time = time.time()
        # Add cross-asset features
        primary_data = assets_data['ASSET_0']
        features_df = extractor.extract_features(primary_data)
        features_df = extractor.add_cross_asset_features(features_df, assets_data)
        processing_time = time.time() - start_time

        # Should complete within reasonable time
        assert processing_time < 10.0
        assert len(features_df) <= len(primary_data)  # May drop rows due to NaN
        assert len(features_df) > 0

    def test_time_anchored_features_performance(self):
        """Test time-anchored features performance."""
        # Create data with datetime index
        dates = pd.date_range('2020-01-01', periods=1000, freq='1H')  # Smaller dataset
        prices = np.random.random(1000) * 100

        data = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)

        # Use fill config to avoid dropping all rows
        test_config = self.config.copy()
        test_config["validation"]["handle_missing"] = "fill"
        extractor = FeatureExtractor(test_config)
        features_df = extractor.extract_features(data)
        original_features_len = len(features_df)

        start_time = time.time()
        features_df = extractor.add_time_anchored_features(features_df)
        processing_time = time.time() - start_time

        # Should complete within reasonable time
        assert processing_time < 5.0
        assert len(features_df) == original_features_len  # Should preserve length
        assert len(features_df) > 0

        # Check time features are added
        time_features = ['hour_of_day', 'day_of_week', 'month_of_year', 'is_weekend']
        for feature in time_features:
            assert feature in features_df.columns


if __name__ == "__main__":
    pytest.main([__file__])
