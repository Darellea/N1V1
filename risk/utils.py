"""
Risk Management Utilities

This module provides shared utility functions for risk management calculations,
including standardized ATR calculations and helper functions.
"""

import logging
from decimal import Decimal
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_divide(
    numerator: Union[float, Decimal],
    denominator: Union[float, Decimal],
    default: Union[float, Decimal] = 0.0,
) -> Union[float, Decimal]:
    """
    Safely divide two numbers, returning a default value if denominator is zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if denominator is zero

    Returns:
        Division result or default value
    """
    try:
        if isinstance(denominator, Decimal) and denominator == 0:
            return default
        elif denominator == 0 or (
            isinstance(denominator, (int, float)) and abs(denominator) < 1e-10
        ):
            return default

        if isinstance(numerator, Decimal) or isinstance(denominator, Decimal):
            return Decimal(str(numerator)) / Decimal(str(denominator))
        else:
            return numerator / denominator
    except (ZeroDivisionError, OverflowError, ValueError):
        return default


def calculate_atr_ema(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    """
    Calculate Average True Range using Exponential Moving Average.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ATR calculation

    Returns:
        ATR value as float, or 0.0 if calculation fails
    """
    try:
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            return 0.0

        # Calculate True Range
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))

        # Handle NaN values that might occur with shift
        hc = hc.fillna(0)
        lc = lc.fillna(0)

        tr = pd.Series(
            np.maximum.reduce([hl.values, hc.values, lc.values]), index=high.index
        )

        # Calculate ATR using EMA
        atr = tr.ewm(span=period).mean().iloc[-1]

        # Handle edge cases
        if np.isnan(atr) or np.isinf(atr) or atr <= 0:
            return 0.0

        return float(atr)

    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return 0.0


def calculate_atr_sma(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    """
    Calculate Average True Range using Simple Moving Average.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ATR calculation

    Returns:
        ATR value as float, or 0.0 if calculation fails
    """
    try:
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            return 0.0

        # Calculate True Range
        hl = high - low
        hc = np.abs(high - close.shift(1))
        lc = np.abs(low - close.shift(1))

        # Handle NaN values that might occur with shift
        hc = hc.fillna(0)
        lc = lc.fillna(0)

        tr = pd.Series(
            np.maximum.reduce([hl.values, hc.values, lc.values]), index=high.index
        )

        # Calculate ATR using SMA
        atr = tr.rolling(window=period).mean().iloc[-1]

        # Handle edge cases
        if np.isnan(atr) or np.isinf(atr) or atr <= 0:
            return 0.0

        return float(atr)

    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return 0.0


def get_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    method: str = "ema",
) -> float:
    """
    Get ATR value using specified method.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Period for ATR calculation
        method: ATR calculation method ('ema' or 'sma')

    Returns:
        ATR value as float
    """
    if method.lower() == "sma":
        return calculate_atr_sma(high, low, close, period)
    else:
        return calculate_atr_ema(high, low, close, period)


def calculate_volatility_percentage(atr: float, current_price: float) -> float:
    """
    Calculate ATR as percentage of current price.

    Args:
        atr: ATR value
        current_price: Current price

    Returns:
        Volatility percentage, or 0.0 if calculation fails
    """
    return float(safe_divide(atr, current_price, 0.0))


def validate_market_data(data: pd.DataFrame) -> bool:
    """
    Validate market data DataFrame for required columns and data quality.

    Args:
        data: Market data DataFrame

    Returns:
        True if data is valid, False otherwise
    """
    if data is None or data.empty:
        return False

    required_columns = ["close"]
    for col in required_columns:
        if col not in data.columns:
            return False
        if data[col].isna().any():
            return False
        if np.isinf(data[col]).any():
            return False
        if (data[col] <= 0).any():
            return False

    return True


def get_config_value(
    config: Dict[str, Any], key: str, default: Any, value_type: type = None
) -> Any:
    """
    Safely get a configuration value with type conversion.

    Args:
        config: Configuration dictionary
        key: Configuration key
        default: Default value
        value_type: Type to convert to (optional)

    Returns:
        Configuration value or default
    """
    try:
        value = config.get(key, default)
        if value_type is not None and value is not None:
            if value_type == Decimal:
                return Decimal(str(value))
            elif value_type == float:
                return float(value)
            elif value_type == int:
                return int(value)
            elif value_type == bool:
                return bool(value)
        return value
    except (ValueError, TypeError):
        return default


def clamp_value(
    value: Union[float, Decimal],
    min_val: Union[float, Decimal],
    max_val: Union[float, Decimal],
) -> Union[float, Decimal]:
    """
    Clamp a value between minimum and maximum values.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


def calculate_z_score(
    series: pd.Series, lookback_period: int
) -> Optional[Dict[str, float]]:
    """
    Calculate z-score for the most recent value in a series.

    This function extracts common z-score calculation logic used across
    different anomaly detectors and risk calculations. By centralizing this
    logic, we eliminate code duplication and ensure consistent statistical
    calculations throughout the risk management system.

    Args:
        series: Time series data (e.g., returns, volumes)
        lookback_period: Number of historical periods to use for mean/std calculation

    Returns:
        Dictionary with z-score calculation results, or None if invalid:
        {
            'z_score': float,  # The calculated z-score
            'current_value': float,  # The most recent value
            'mean': float,  # Historical mean
            'std': float  # Historical standard deviation
        }
    """
    try:
        if len(series) < lookback_period + 1:
            return None

        # Get recent data for calculation
        recent_data = series.tail(lookback_period)
        current_value = float(recent_data.iloc[-1])

        # Calculate historical statistics (excluding current value)
        historical_data = recent_data.iloc[:-1]
        mean_val = historical_data.mean()
        std_val = historical_data.std()

        # Handle edge cases
        if std_val == 0 or np.isnan(std_val) or np.isnan(mean_val):
            return None

        # Calculate z-score
        z_score = (current_value - mean_val) / std_val

        return {
            "z_score": float(z_score),
            "current_value": current_value,
            "mean": mean_val,
            "std": std_val,
        }

    except Exception as e:
        logger.warning(f"Error calculating z-score: {e}")
        return None


def calculate_returns(prices: pd.Series) -> Optional[pd.Series]:
    """
    Calculate price returns from a price series.

    This function provides standardized return calculation used in
    volatility and anomaly detection calculations.

    Args:
        prices: Price series

    Returns:
        Returns series, or None if calculation fails
    """
    try:
        if len(prices) < 2:
            return None

        returns = prices.pct_change().dropna()

        if returns.empty:
            return None

        return returns

    except Exception as e:
        logger.warning(f"Error calculating returns: {e}")
        return None


def enhanced_validate_market_data(
    data: pd.DataFrame, required_columns: Optional[list] = None
) -> bool:
    """
    Enhanced market data validation with configurable required columns.

    This function consolidates market data validation logic used across
    different risk management components, ensuring consistent data quality
    checks and reducing code duplication.

    Args:
        data: Market data DataFrame
        required_columns: List of required column names (defaults to ['close'])

    Returns:
        True if data is valid, False otherwise
    """
    if data is None or data.empty:
        return False

    # Default required columns
    if required_columns is None:
        required_columns = ["close"]

    # Check for required columns
    for col in required_columns:
        if col not in data.columns:
            return False

        # Validate data quality
        if data[col].isna().any():
            return False

        if np.isinf(data[col]).any():
            return False

        # Check for non-positive values in price columns
        if col in ["open", "high", "low", "close"]:
            if (data[col] <= 0).any():
                return False

    return True
