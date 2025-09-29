"""
Technical Indicators Module

This module provides implementations of various technical indicators used in trading strategies.
All indicators are designed to work with pandas DataFrames containing OHLCV data.

Supported indicators:
- RSI (Relative Strength Index)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- ADX (Average Directional Index)
- OBV (On-Balance Volume)
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Centralized configuration for indicator parameters
# This eliminates hard-coded values and allows easy configuration changes
INDICATOR_CONFIG = {
    "rsi_period": 14,
    "ema_period": 20,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std_dev": 2.0,
    "atr_period": 14,
    "adx_period": 14,
}


def calculate_rsi(
    data: pd.DataFrame, period: int = 14, column: str = "close"
) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data: DataFrame with OHLCV data
        period: RSI calculation period
        column: Column to use for calculation (default: 'close')

    Returns:
        Series with RSI values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")

    if len(data) < period:
        logger.warning(f"Insufficient data for RSI calculation: {len(data)} < {period}")
        return pd.Series([np.nan] * len(data), index=data.index)

    delta = data[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(
    data: pd.DataFrame, period: int = 20, column: str = "close"
) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    Args:
        data: DataFrame with OHLCV data
        period: EMA period
        column: Column to use for calculation (default: 'close')

    Returns:
        Series with EMA values
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")

    if len(data) < period:
        logger.warning(f"Insufficient data for EMA calculation: {len(data)} < {period}")
        return pd.Series([np.nan] * len(data), index=data.index)

    return data[column].ewm(span=period, adjust=False).mean()


def calculate_macd(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = "close",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        data: DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period
        column: Column to use for calculation (default: 'close')

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")

    min_period = max(fast_period, slow_period, signal_period)
    if len(data) < min_period:
        logger.warning(
            f"Insufficient data for MACD calculation: {len(data)} < {min_period}"
        )
        nan_series = pd.Series([np.nan] * len(data), index=data.index)
        return nan_series, nan_series, nan_series

    fast_ema = data[column].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data[column].ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, column: str = "close"
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        data: DataFrame with OHLCV data
        period: Moving average period
        std_dev: Standard deviation multiplier
        column: Column to use for calculation (default: 'close')

    Returns:
        Tuple of (Upper Band, Middle Band/SMA, Lower Band)
    """
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data")

    if len(data) < period:
        logger.warning(
            f"Insufficient data for Bollinger Bands calculation: {len(data)} < {period}"
        )
        nan_series = pd.Series([np.nan] * len(data), index=data.index)
        return nan_series, nan_series, nan_series

    sma = data[column].rolling(window=period).mean()
    std = data[column].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    return upper_band, sma, lower_band


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        data: DataFrame with OHLCV data (must contain 'high', 'low', 'close')
        period: ATR calculation period

    Returns:
        Series with ATR values
    """
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    if len(data) < period:
        logger.warning(f"Insufficient data for ATR calculation: {len(data)} < {period}")
        return pd.Series([np.nan] * len(data), index=data.index)

    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift(1)).abs()
    low_close = (data["low"] - data["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).

    Args:
        data: DataFrame with OHLCV data (must contain 'high', 'low', 'close')
        period: ADX calculation period

    Returns:
        Series with ADX values
    """
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    if len(data) < period * 2:
        logger.warning(
            f"Insufficient data for ADX calculation: {len(data)} < {period * 2}"
        )
        return pd.Series([np.nan] * len(data), index=data.index)

    # Calculate directional movements
    delta_high = data["high"].diff()
    delta_low = data["low"].diff()

    plus_dm = np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0)
    minus_dm = np.where((delta_low > delta_high) & (delta_low > 0), delta_low, 0)

    # Calculate ATR for normalization
    atr = calculate_atr(data, period)

    # Calculate directional indicators
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

    # Calculate DX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    # Calculate ADX
    adx = dx.rolling(window=period).mean()

    return adx


def calculate_obv(data: pd.DataFrame) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).

    Args:
        data: DataFrame with OHLCV data (must contain 'close', 'volume')

    Returns:
        Series with OBV values
    """
    required_cols = ["close", "volume"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    if len(data) < 2:
        logger.warning(
            "Insufficient data for OBV calculation: need at least 2 data points"
        )
        return pd.Series([np.nan] * len(data), index=data.index)

    obv = pd.Series([0] * len(data), index=data.index, dtype=int)
    obv.iloc[0] = int(data["volume"].iloc[0])

    for i in range(1, len(data)):
        if data["close"].iloc[i] > data["close"].iloc[i - 1]:
            obv.iloc[i] = int(obv.iloc[i - 1] + data["volume"].iloc[i])
        elif data["close"].iloc[i] < data["close"].iloc[i - 1]:
            obv.iloc[i] = int(obv.iloc[i - 1] - data["volume"].iloc[i])
        else:
            obv.iloc[i] = int(obv.iloc[i - 1])

    return obv


def calculate_stochastic(
    data: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (%K and %D).

    Args:
        data: DataFrame with OHLCV data (must contain 'high', 'low', 'close')
        k_period: Period for %K calculation
        d_period: Period for %D calculation (SMA of %K)

    Returns:
        Tuple of (%K, %D) Series
    """
    required_cols = ["high", "low", "close"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    if len(data) < k_period:
        logger.warning(
            f"Insufficient data for Stochastic calculation: {len(data)} < {k_period}"
        )
        nan_series = pd.Series([np.nan] * len(data), index=data.index)
        return nan_series, nan_series

    # Calculate %K
    lowest_low = data["low"].rolling(window=k_period).min()
    highest_high = data["high"].rolling(window=k_period).max()

    k_percent = 100 * (data["close"] - lowest_low) / (highest_high - lowest_low)

    # Calculate %D (SMA of %K)
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        data: DataFrame with OHLCV data (must contain 'high', 'low', 'close', 'volume')

    Returns:
        Series with VWAP values
    """
    required_cols = ["high", "low", "close", "volume"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    if len(data) < 1:
        logger.warning("Insufficient data for VWAP calculation")
        return pd.Series([np.nan] * len(data), index=data.index)

    # VWAP = (Cumulative (High + Low + Close)/3 * Volume) / Cumulative Volume
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    cumulative_tp_volume = (typical_price * data["volume"]).cumsum()
    cumulative_volume = data["volume"].cumsum()

    vwap = cumulative_tp_volume / cumulative_volume

    return vwap


def calculate_all_indicators(
    data: pd.DataFrame, config: Optional[Dict[str, Union[int, float]]] = None
) -> pd.DataFrame:
    """
    Calculate all supported technical indicators.

    Args:
        data: DataFrame with OHLCV data
        config: Optional configuration dictionary for indicator parameters

    Returns:
        DataFrame with all indicator columns added
    """
    # Use centralized config as defaults, overridden by passed config
    # This centralizes configuration and eliminates hard-coded values
    params = {**INDICATOR_CONFIG, **(config or {})}

    df = data.copy()

    try:
        # RSI
        df["rsi"] = calculate_rsi(df, period=params["rsi_period"])

        # EMA
        df["ema"] = calculate_ema(df, period=params["ema_period"])

        # MACD
        macd_line, signal_line, histogram = calculate_macd(
            df,
            fast_period=params["macd_fast"],
            slow_period=params["macd_slow"],
            signal_period=params["macd_signal"],
        )
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram

        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(
            df, period=params["bb_period"], std_dev=params["bb_std_dev"]
        )
        df["bb_upper"] = upper_bb
        df["bb_middle"] = middle_bb
        df["bb_lower"] = lower_bb

        # ATR
        df["atr"] = calculate_atr(df, period=params["atr_period"])

        # ADX
        df["adx"] = calculate_adx(df, period=params["adx_period"])

        # OBV
        df["obv"] = calculate_obv(df)

        logger.info(
            f"Calculated {len(df.columns) - len(data.columns)} indicator columns"
        )

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        # Return original data if calculation fails
        return data

    return df


def get_indicator_names() -> list:
    """
    Get list of all indicator column names that are added by calculate_all_indicators.

    Returns:
        List of indicator column names
    """
    return [
        "rsi",
        "ema",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "atr",
        "adx",
        "obv",
    ]


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """
    Validate that DataFrame contains required OHLCV columns.

    Args:
        data: DataFrame to validate

    Returns:
        True if valid, False otherwise
    """
    required_cols = ["open", "high", "low", "close", "volume"]
    return all(col in data.columns for col in required_cols)
