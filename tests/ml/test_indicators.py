"""
Unit tests for indicators module.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.indicators import (
    calculate_adx,
    calculate_all_indicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    get_indicator_names,
    validate_ohlcv_data,
)


class TestRSI:
    """Test RSI calculations."""

    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation."""
        # Create sample data
        data = pd.DataFrame(
            {"close": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14]}
        )

        rsi = calculate_rsi(data, period=14)
        assert len(rsi) == len(data)
        assert not rsi.isna().all()
        assert 0 <= rsi.iloc[-1] <= 100

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        data = pd.DataFrame({"close": [10, 11, 12]})
        rsi = calculate_rsi(data, period=14)
        assert rsi.isna().all()

    def test_calculate_rsi_invalid_column(self):
        """Test RSI with invalid column."""
        data = pd.DataFrame({"price": [10, 11, 12]})
        with pytest.raises(ValueError, match="Column 'close' not found"):
            calculate_rsi(data)


class TestEMA:
    """Test EMA calculations."""

    def test_calculate_ema_basic(self):
        """Test basic EMA calculation."""
        data = pd.DataFrame(
            {
                "close": [
                    10,
                    11,
                    12,
                    11,
                    10,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    15,
                    14,
                    13,
                    12,
                    11,
                    10,
                    11,
                ]
            }
        )

        ema = calculate_ema(data, period=10)
        assert len(ema) == len(data)
        assert not ema.isna().all()
        # EMA starts calculating immediately, so first value should not be NaN
        assert not pd.isna(ema.iloc[0])
        assert not pd.isna(ema.iloc[-1])

    def test_calculate_ema_insufficient_data(self):
        """Test EMA with insufficient data."""
        data = pd.DataFrame({"close": [10, 11, 12]})
        ema = calculate_ema(data, period=10)
        assert ema.isna().all()


class TestMACD:
    """Test MACD calculations."""

    def test_calculate_macd_basic(self):
        """Test basic MACD calculation."""
        data = pd.DataFrame({"close": list(range(10, 50))})  # 40 data points

        macd_line, signal_line, histogram = calculate_macd(data)
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
        assert not macd_line.isna().all()
        assert not signal_line.isna().all()
        assert not histogram.isna().all()

    def test_calculate_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        data = pd.DataFrame({"close": [10, 11, 12]})
        macd_line, signal_line, histogram = calculate_macd(data)
        assert macd_line.isna().all()
        assert signal_line.isna().all()
        assert histogram.isna().all()


class TestBollingerBands:
    """Test Bollinger Bands calculations."""

    def test_calculate_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation."""
        data = pd.DataFrame(
            {"close": [10] * 30}  # Constant price for predictable results
        )

        upper, middle, lower = calculate_bollinger_bands(data, period=20)
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)
        assert not upper.isna().all()
        assert not middle.isna().all()
        assert not lower.isna().all()

        # For constant data, bands should be at the price level
        assert upper.iloc[-1] == 10.0
        assert middle.iloc[-1] == 10.0
        assert lower.iloc[-1] == 10.0

    def test_calculate_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data."""
        data = pd.DataFrame({"close": [10, 11, 12]})
        upper, middle, lower = calculate_bollinger_bands(data, period=20)
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()


class TestATR:
    """Test ATR calculations."""

    def test_calculate_atr_basic(self):
        """Test basic ATR calculation."""
        data = pd.DataFrame(
            {
                "high": [12, 13, 14, 13, 12, 11, 12, 13, 14, 15, 16, 17, 18, 17, 16],
                "low": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 16, 15, 14],
                "close": [11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 15, 16, 17, 16, 15],
            }
        )

        atr = calculate_atr(data, period=14)
        assert len(atr) == len(data)
        assert not atr.isna().all()
        assert atr.iloc[-1] >= 0

    def test_calculate_atr_missing_columns(self):
        """Test ATR with missing required columns."""
        data = pd.DataFrame({"close": [10, 11, 12]})
        with pytest.raises(ValueError, match="Column 'high' not found"):
            calculate_atr(data)


class TestADX:
    """Test ADX calculations."""

    def test_calculate_adx_basic(self):
        """Test basic ADX calculation."""
        # Create trending data with proper directional movements
        np.random.seed(42)
        base = np.linspace(10, 50, 60)
        noise = np.random.normal(0, 0.5, 60)

        data = pd.DataFrame(
            {
                "high": base + 2 + noise,  # Upward trend
                "low": base - 2 + noise,
                "close": base + noise,
            }
        )

        adx = calculate_adx(data, period=14)
        assert len(adx) == len(data)
        # ADX may have NaN values at the beginning, but should have some valid values
        assert not adx.iloc[28:].isna().all()  # After sufficient data
        if not pd.isna(adx.iloc[-1]):
            assert 0 <= adx.iloc[-1] <= 100

    def test_calculate_adx_insufficient_data(self):
        """Test ADX with insufficient data."""
        data = pd.DataFrame(
            {"high": [10, 11, 12], "low": [8, 9, 10], "close": [9, 10, 11]}
        )
        adx = calculate_adx(data, period=14)
        assert adx.isna().all()


class TestOBV:
    """Test OBV calculations."""

    def test_calculate_obv_basic(self):
        """Test basic OBV calculation."""
        data = pd.DataFrame(
            {
                "close": [10, 11, 12, 11, 10, 9, 10, 11, 12, 13],
                "volume": [100, 110, 120, 90, 80, 70, 100, 110, 120, 130],
            }
        )

        obv = calculate_obv(data)
        assert len(obv) == len(data)
        assert not obv.isna().all()
        assert obv.iloc[0] == 100  # First value should equal first volume

    def test_calculate_obv_single_row(self):
        """Test OBV with single row data (should return volume value)."""
        data = pd.DataFrame({"close": [10], "volume": [100]})
        obv = calculate_obv(data)
        assert not obv.isna().all()
        assert obv.iloc[0] == 100  # Single row should return volume value


class TestAllIndicators:
    """Test calculate_all_indicators function."""

    def test_calculate_all_indicators_basic(self):
        """Test calculating all indicators."""
        data = pd.DataFrame(
            {
                "open": list(range(10, 50)),
                "high": list(range(12, 52)),
                "low": list(range(8, 48)),
                "close": list(range(11, 51)),
                "volume": [100] * 40,
            }
        )

        result = calculate_all_indicators(data)
        assert len(result.columns) > len(data.columns)

        # Check that all expected indicators are present
        expected_indicators = get_indicator_names()
        for indicator in expected_indicators:
            assert indicator in result.columns

    def test_calculate_all_indicators_with_config(self):
        """Test calculating all indicators with custom config."""
        data = pd.DataFrame(
            {
                "open": list(range(10, 50)),
                "high": list(range(12, 52)),
                "low": list(range(8, 48)),
                "close": list(range(11, 51)),
                "volume": [100] * 40,
            }
        )

        config = {"rsi_period": 10, "ema_period": 5, "bb_period": 10}

        result = calculate_all_indicators(data, config)
        assert len(result.columns) > len(data.columns)

    def test_calculate_all_indicators_insufficient_data(self):
        """Test calculating all indicators with insufficient data."""
        data = pd.DataFrame(
            {
                "open": [10, 11],
                "high": [12, 13],
                "low": [8, 9],
                "close": [11, 12],
                "volume": [100, 110],
            }
        )

        result = calculate_all_indicators(data)
        # Should add indicator columns even with insufficient data (filled with NaN)
        assert len(result.columns) > len(data.columns)
        expected_indicators = get_indicator_names()
        for indicator in expected_indicators:
            assert indicator in result.columns


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_indicator_names(self):
        """Test get_indicator_names function."""
        names = get_indicator_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "rsi" in names
        assert "ema" in names
        assert "macd" in names

    def test_validate_ohlcv_data_valid(self):
        """Test validate_ohlcv_data with valid data."""
        data = pd.DataFrame(
            {
                "open": [10, 11],
                "high": [12, 13],
                "low": [8, 9],
                "close": [11, 12],
                "volume": [100, 110],
            }
        )
        assert validate_ohlcv_data(data) is True

    def test_validate_ohlcv_data_invalid(self):
        """Test validate_ohlcv_data with invalid data."""
        # Missing volume
        data = pd.DataFrame(
            {"open": [10, 11], "high": [12, 13], "low": [8, 9], "close": [11, 12]}
        )
        assert validate_ohlcv_data(data) is False

        # Missing high
        data = pd.DataFrame(
            {"open": [10, 11], "low": [8, 9], "close": [11, 12], "volume": [100, 110]}
        )
        assert validate_ohlcv_data(data) is False


class TestErrorHandling:
    """Test error handling in indicators."""

    @patch("ml.indicators.logger")
    def test_calculate_rsi_with_exception(self, mock_logger):
        """Test RSI error handling."""
        data = pd.DataFrame({"close": []})  # Empty data

        rsi = calculate_rsi(data, period=14)
        assert len(rsi) == 0

    @patch("ml.indicators.logger")
    def test_calculate_all_indicators_with_exception(self, mock_logger):
        """Test calculate_all_indicators error handling."""
        data = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []}
        )

        result = calculate_all_indicators(data)
        # With empty data, indicators are calculated but result in empty series
        # So we get the original columns plus indicator columns
        assert len(result.columns) > len(data.columns)
        expected_indicators = get_indicator_names()
        for indicator in expected_indicators:
            assert indicator in result.columns
