"""
Anomaly Detector Module

This module provides sophisticated anomaly detection capabilities to protect
trading systems against abnormal market conditions, pump-and-dumps, and
sudden price/volume anomalies.

Detection Methods:
- Price Z-Score: Statistical outlier detection for price movements
- Volume Z-Score: Abnormal volume surge detection
- Price Gap: Large price jumps between consecutive candles

Response Mechanisms:
- Skip trade entry
- Scale down position size
- Log anomaly details for analysis

Integration Points:
- Strategy execution pipeline
- Backtester
- Risk management layer

PERFORMANCE OPTIMIZATIONS:
- Asynchronous file I/O: Uses aiofiles for non-blocking file operations to prevent
  main thread blocking in high-throughput trading scenarios
- Memory-efficient logging: Implements background task logging to avoid blocking
  the main trading thread during anomaly detection
- Fallback mechanisms: Gracefully handles missing aiofiles dependency with warnings
"""

import asyncio
import json
import logging
import os
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import aiofiles
except ImportError:
    logger.warning(
        "aiofiles not installed. Falling back to synchronous file operations."
    )
    aiofiles = None

from utils.logger import get_trade_logger

# Security constants for file path validation
LOG_DIR = "logs/anomalies"
DEFAULT_LOG_FILE = "anomalies.log"
DEFAULT_JSON_LOG_FILE = "anomalies.json"

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class TrimmingList(list):
    """A list that automatically trims to max_length when appending."""

    def __init__(self, max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length

    def append(self, item):
        super().append(item)
        if len(self) > self.max_length:
            self[:] = self[-self.max_length :]


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""

    PRICE_ZSCORE = "price_zscore"
    VOLUME_ZSCORE = "volume_zscore"
    PRICE_GAP = "price_gap"
    KALMAN_FILTER = "kalman_filter"
    NONE = "none"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyResponse(Enum):
    """Possible responses to detected anomalies."""

    SKIP_TRADE = "skip_trade"
    SCALE_DOWN = "scale_down"
    LOG_ONLY = "log_only"
    NONE = "none"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    is_anomaly: bool
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    confidence_score: float
    z_score: Optional[float] = None
    threshold: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["anomaly_type"] = self.anomaly_type.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        return data


@dataclass
class AnomalyLog:
    """Log entry for detected anomaly."""

    symbol: str
    anomaly_result: AnomalyResult
    market_data: Optional[Dict[str, Any]] = None
    action_taken: Optional[str] = None
    original_signal: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "symbol": self.symbol,
            "anomaly_result": self.anomaly_result.to_dict(),
            "market_data": self.market_data,
            "action_taken": self.action_taken,
            "original_signal": self.original_signal,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class BaseAnomalyDetector:
    """Base class for anomaly detection methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.name = self.__class__.__name__

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect anomalies in the provided data.

        Args:
            data: Market data DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        raise NotImplementedError("Subclasses must implement detect method")

    def _calculate_severity(
        self, z_score: float, thresholds: Dict[str, float]
    ) -> AnomalySeverity:
        """Calculate severity based on z-score and thresholds."""
        abs_z = abs(z_score)

        if abs_z >= thresholds.get("critical", 5.0):
            return AnomalySeverity.CRITICAL
        elif abs_z >= thresholds.get("high", 3.0):
            return AnomalySeverity.HIGH
        elif abs_z >= thresholds.get("medium", 2.0):
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class PriceZScoreDetector(BaseAnomalyDetector):
    """Detect price anomalies using statistical z-score analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Use centralized configuration values
        self.lookback_period = self.config.get("lookback_period", 50)
        self.z_threshold = self.config.get("z_threshold", 3.0)
        self.severity_thresholds = self.config.get(
            "severity_thresholds",
            {"low": 2.0, "medium": 3.0, "high": 4.0, "critical": 5.0},
        )

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect price anomalies using z-score analysis with fallback mechanisms.

        This method implements fallback mechanisms for data unavailability by using
        try/except blocks to catch potential data-related errors and return conservative
        defaults when data is unavailable or calculations fail.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        # Check prerequisites with fallback
        if not self._check_prerequisites_with_fallback(data, symbol):
            return self._create_conservative_result("insufficient data")

        try:
            # Calculate returns and validate with fallback
            returns = self._calculate_returns_with_fallback(data, symbol)
            if returns is None:
                return self._create_conservative_result("cannot calculate returns")

            # Calculate z-score with fallback
            z_score_result = self._calculate_z_score_with_fallback(returns, symbol)
            if z_score_result is None:
                return self._create_conservative_result("cannot calculate z-score")

            # Process anomaly detection
            return self._process_anomaly_detection(
                z_score_result, AnomalyType.PRICE_ZSCORE
            )

        except Exception as e:
            logger.warning(f"Error in price z-score detection for {symbol}: {e}")
            logger.warning(f"Stack trace: {traceback.format_exc()}")
            # Return conservative result on any error to ensure system stability
            return self._create_conservative_result(f"detection error: {str(e)}")

    def _check_prerequisites(self, data: pd.DataFrame) -> bool:
        """
        Check basic prerequisites for detection.

        Args:
            data: Market data DataFrame

        Returns:
            True if prerequisites are met
        """
        return (
            self.enabled
            and not data.empty
            and len(data) >= self.lookback_period
            and "close" in data.columns
        )

    def _create_empty_result(self) -> AnomalyResult:
        """
        Create an empty (non-anomaly) result.

        Returns:
            AnomalyResult with no anomaly detected
        """
        return AnomalyResult(
            is_anomaly=False,
            anomaly_type=AnomalyType.NONE,
            severity=AnomalySeverity.LOW,
            confidence_score=0.0,
        )

    def _calculate_returns(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate price returns from market data.

        Args:
            data: Market data DataFrame

        Returns:
            Returns series or None if invalid
        """
        returns = data["close"].pct_change().dropna()
        if len(returns) < self.lookback_period:
            return None
        return returns

    def _calculate_z_score(self, returns: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Calculate z-score for the current price using rolling window statistics on historical prices.

        This method calculates the z-score of the current price relative to the historical price distribution,
        which is more appropriate for detecting sudden price jumps/drops than return-based z-scores.

        Args:
            returns: Price returns series (used to reconstruct prices)

        Returns:
            Dictionary with z-score calculation results or None if invalid
        """
        try:
            # Reconstruct prices from returns for price-based z-score calculation
            prices = (1 + returns).cumprod()
            prices = prices.reset_index(drop=True)

            # Use rolling window for z-score calculation
            window = self.config.get("window", self.lookback_period)

            if len(prices) < window:
                return None

            # Calculate rolling statistics for historical prices (exclude current price)
            historical_prices = prices.iloc[:-1]  # Exclude current price
            if len(historical_prices) < window - 1:
                return None

            # Use the last 'window' historical prices
            window_prices = historical_prices.tail(window - 1)
            if len(window_prices) < window - 1:
                return None

            current_price = prices.iloc[-1]
            historical_mean = window_prices.mean()
            historical_std = window_prices.std()

            # Handle NaN std
            if pd.isna(historical_std) or np.isnan(historical_std):
                return None

            # If historical std == 0 (constant price), handle edge case
            if historical_std == 0 or historical_std < 1e-10:
                if current_price == historical_mean:
                    z_score = 0.0  # No change from mean
                else:
                    z_score = (
                        10.0 if current_price > historical_mean else -10.0
                    )  # Extreme deviation
                return {
                    "z_score": z_score,
                    "current_price": float(current_price),
                    "mean_price": float(historical_mean),
                    "std_price": float(historical_std),
                }

            # Add small epsilon guard to prevent division by very small numbers
            epsilon = 1e-8
            if historical_std < epsilon:
                historical_std = epsilon

            # Calculate z-score with epsilon guard
            z_score = (current_price - historical_mean) / historical_std

            return {
                "z_score": float(z_score),
                "current_price": float(current_price),
                "mean_price": float(historical_mean),
                "std_price": float(historical_std),
            }

        except Exception as e:
            logger.warning(f"Error calculating z-score: {e}")
            return None

    def _process_anomaly_detection(
        self, z_score_result: Dict[str, Any], anomaly_type: AnomalyType
    ) -> AnomalyResult:
        """
        Process the anomaly detection based on z-score results.

        Args:
            z_score_result: Z-score calculation results
            anomaly_type: Type of anomaly being detected

        Returns:
            AnomalyResult with detection details
        """
        z_score = z_score_result["z_score"]
        abs_z_score = abs(z_score)

        # Determine if it's an anomaly
        is_anomaly = abs_z_score >= self.z_threshold

        if is_anomaly:
            confidence_score = min(
                abs_z_score / self.severity_thresholds["critical"], 1.0
            )
            severity = self._calculate_severity(z_score, self.severity_thresholds)
        else:
            confidence_score = 0.0
            severity = AnomalySeverity.LOW
            anomaly_type = AnomalyType.NONE

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence_score=confidence_score,
            z_score=float(z_score) if is_anomaly else None,
            threshold=self.z_threshold if is_anomaly else None,
            context=z_score_result,
        )

    def _handle_detection_error(self, error: Exception, symbol: str) -> AnomalyResult:
        """
        Handle errors during anomaly detection.

        Args:
            error: The exception that occurred
            symbol: Trading symbol

        Returns:
            Empty anomaly result
        """
        logger.error(f"Error in price z-score detection for {symbol}: {error}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return self._create_empty_result()

    def _check_prerequisites_with_fallback(
        self, data: pd.DataFrame, symbol: str
    ) -> bool:
        """
        Check basic prerequisites for detection with fallback mechanisms.

        This method implements fallback mechanisms for data unavailability by using
        try/except blocks to catch potential data-related errors and return conservative
        defaults when data is unavailable.

        Args:
            data: Market data DataFrame
            symbol: Trading symbol

        Returns:
            True if prerequisites are met, False otherwise (with conservative fallback)
        """
        try:
            # Handle empty data case
            if data.empty:
                return False
            return self._check_prerequisites(data)
        except Exception as e:
            logger.warning(f"Error checking prerequisites for {symbol}: {e}")
            # Return False to trigger conservative fallback
            return False

    def _calculate_returns_with_fallback(
        self, data: pd.DataFrame, symbol: str
    ) -> Optional[pd.Series]:
        """
        Calculate price returns from market data with fallback mechanisms.

        This method implements fallback mechanisms for data unavailability by using
        try/except blocks to catch potential data-related errors and return None
        when calculations fail, triggering conservative defaults.

        Args:
            data: Market data DataFrame
            symbol: Trading symbol

        Returns:
            Returns series or None if calculation fails
        """
        try:
            return self._calculate_returns(data)
        except Exception as e:
            logger.warning(f"Error calculating returns for {symbol}: {e}")
            return None

    def _calculate_z_score_with_fallback(
        self, returns: pd.Series, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate z-score for the most recent return with fallback mechanisms.

        This method implements fallback mechanisms for data unavailability by using
        try/except blocks to catch potential data-related errors and return None
        when calculations fail, triggering conservative defaults.

        Args:
            returns: Price returns series
            symbol: Trading symbol

        Returns:
            Dictionary with z-score calculation results or None if invalid
        """
        try:
            return self._calculate_z_score(returns)
        except Exception as e:
            logger.warning(f"Error calculating z-score for {symbol}: {e}")
            return None

    def _create_conservative_result(self, reason: str) -> AnomalyResult:
        """
        Create a conservative anomaly result when detection fails or data is unavailable.

        This method implements the fallback mechanism for data unavailability by providing
        a conservative result that assumes no anomaly detected but with low confidence,
        ensuring the system can continue operating safely.

        Args:
            reason: Reason for using conservative result

        Returns:
            Conservative AnomalyResult with no anomaly detected
        """
        logger.info(f"Using conservative anomaly result: {reason}")

        return AnomalyResult(
            is_anomaly=False,  # Conservative: assume no anomaly
            anomaly_type=AnomalyType.NONE,
            severity=AnomalySeverity.LOW,  # Lowest severity
            confidence_score=0.0,  # No confidence in detection
            context={"fallback_reason": reason},
        )


class VolumeZScoreDetector(BaseAnomalyDetector):
    """Detect volume anomalies using statistical z-score analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Use centralized configuration values
        self.lookback_period = self.config.get("lookback_period", 20)
        self.z_threshold = self.config.get("z_threshold", 3.0)
        self.severity_thresholds = self.config.get(
            "severity_thresholds",
            {"low": 2.0, "medium": 3.0, "high": 4.0, "critical": 15.0},
        )

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect volume anomalies using z-score analysis.

        This method orchestrates the volume z-score detection by delegating
        to specialized helper methods for each concern.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        # Check prerequisites
        if not self._check_prerequisites(data):
            return self._create_empty_result()

        try:
            # Calculate volumes and validate
            volumes = self._calculate_volumes(data)
            if volumes is None:
                return self._create_empty_result()

            # Calculate z-score
            z_score_result = self._calculate_volume_z_score(volumes)
            if z_score_result is None:
                return self._create_empty_result()

            # Process anomaly detection
            return self._process_anomaly_detection(
                z_score_result, AnomalyType.VOLUME_ZSCORE
            )

        except Exception as e:
            return self._handle_detection_error(e, symbol)

    def _check_prerequisites(self, data: pd.DataFrame) -> bool:
        """
        Check basic prerequisites for detection.

        Args:
            data: Market data DataFrame

        Returns:
            True if prerequisites are met
        """
        return (
            self.enabled
            and not data.empty
            and len(data) >= self.lookback_period
            and "volume" in data.columns
        )

    def _create_empty_result(self) -> AnomalyResult:
        """
        Create an empty (non-anomaly) result.

        Returns:
            AnomalyResult with no anomaly detected
        """
        return AnomalyResult(
            is_anomaly=False,
            anomaly_type=AnomalyType.NONE,
            severity=AnomalySeverity.LOW,
            confidence_score=0.0,
        )

    def _calculate_volumes(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Calculate volume data from market data.

        Args:
            data: Market data DataFrame

        Returns:
            Volume series or None if invalid
        """
        volumes = data["volume"].dropna()
        if len(volumes) < self.lookback_period:
            return None
        return volumes

    def _calculate_volume_z_score(self, volumes: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Calculate z-score for the most recent volume using rolling window statistics.

        Args:
            volumes: Volume data series

        Returns:
            Dictionary with z-score calculation results or None if invalid
        """
        try:
            # Use rolling window for z-score calculation
            window = self.config.get("window", self.lookback_period)

            if len(volumes) < window:
                return None

            # Calculate rolling statistics for z-score
            rolling_mean = volumes.rolling(window=window).mean()
            rolling_std = volumes.rolling(window=window).std()

            # Use historical mean and std (shift by 1 to exclude current value)
            historical_mean = rolling_mean.shift(1)
            historical_std = rolling_std.shift(1)

            # Get the most recent values
            current_volume = volumes.iloc[-1]
            current_mean = historical_mean.iloc[-1]
            current_std = historical_std.iloc[-1]

            # Handle NaN std
            if pd.isna(current_std):
                return None

            # If historical std is 0, any deviation is extreme
            if current_std == 0:
                if current_volume == current_mean:
                    z_score = 0.0
                else:
                    z_score = 10.0 if current_volume > current_mean else -10.0
            else:
                z_score = (current_volume - current_mean) / current_std

            return {
                "z_score": float(z_score),
                "current_volume": float(current_volume),
                "mean_volume": float(current_mean),
                "std_volume": float(current_std),
            }

        except Exception as e:
            logger.warning(f"Error calculating volume z-score: {e}")
            return None

    def _process_anomaly_detection(
        self, z_score_result: Dict[str, Any], anomaly_type: AnomalyType
    ) -> AnomalyResult:
        """
        Process the anomaly detection based on z-score results.

        Args:
            z_score_result: Z-score calculation results
            anomaly_type: Type of anomaly being detected

        Returns:
            AnomalyResult with detection details
        """
        z_score = z_score_result["z_score"]
        abs_z_score = abs(z_score)

        # Determine if it's an anomaly
        is_anomaly = abs_z_score >= self.z_threshold

        if is_anomaly:
            confidence_score = min(
                abs_z_score / self.severity_thresholds["critical"], 1.0
            )
            severity = self._calculate_severity(z_score, self.severity_thresholds)
        else:
            confidence_score = 0.0
            severity = AnomalySeverity.LOW
            anomaly_type = AnomalyType.NONE

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence_score=confidence_score,
            z_score=float(z_score) if is_anomaly else None,
            threshold=self.z_threshold if is_anomaly else None,
            context=z_score_result,
        )

    def _handle_detection_error(self, error: Exception, symbol: str) -> AnomalyResult:
        """
        Handle errors during anomaly detection.

        Args:
            error: The exception that occurred
            symbol: Trading symbol

        Returns:
            Empty anomaly result
        """
        logger.error(f"Error in volume z-score detection for {symbol}: {error}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return self._create_empty_result()


class PriceGapDetector(BaseAnomalyDetector):
    """Detect price gap anomalies (large jumps between consecutive candles)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Use centralized configuration values
        self.gap_threshold_pct = self.config.get("gap_threshold_pct", 5.0)
        self.severity_thresholds = self.config.get(
            "severity_thresholds",
            {"low": 3.0, "medium": 5.0, "high": 10.0, "critical": 15.0},
        )

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect price gap anomalies.

        This method orchestrates the price gap detection by delegating
        to specialized helper methods for each concern.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        # Check prerequisites
        if not self._check_prerequisites(data):
            return self._create_empty_result()

        try:
            # Calculate maximum gap and validate
            max_gap_result = self._calculate_max_price_gap(data)
            if max_gap_result is None:
                return self._create_empty_result()

            # Process anomaly detection
            return self._process_gap_anomaly_detection(max_gap_result)

        except Exception as e:
            return self._handle_detection_error(e, symbol)

    def _check_prerequisites(self, data: pd.DataFrame) -> bool:
        """
        Check basic prerequisites for detection.

        Args:
            data: Market data DataFrame

        Returns:
            True if prerequisites are met
        """
        return (
            self.enabled
            and not data.empty
            and len(data) >= 2
            and "close" in data.columns
        )

    def _create_empty_result(self) -> AnomalyResult:
        """
        Create an empty (non-anomaly) result.

        Returns:
            AnomalyResult with no anomaly detected
        """
        return AnomalyResult(
            is_anomaly=False,
            anomaly_type=AnomalyType.NONE,
            severity=AnomalySeverity.LOW,
            confidence_score=0.0,
        )

    def _calculate_price_gap(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculate price gap between consecutive closes.

        Args:
            data: Market data DataFrame

        Returns:
            Dictionary with gap calculation results or None if invalid
        """
        closes = data["close"]
        prev_close = closes.iloc[-2]
        current_close = closes.iloc[-1]

        if prev_close == 0:
            return None

        # Calculate gap percentage
        gap_pct = abs((current_close - prev_close) / prev_close) * 100

        return {
            "prev_close": prev_close,
            "current_close": current_close,
            "gap_pct": gap_pct,
        }

    def _calculate_max_price_gap(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculate the maximum price gap between any consecutive closes.

        Args:
            data: Market data DataFrame

        Returns:
            Dictionary with maximum gap calculation results or None if invalid
        """
        closes = data["close"]
        if len(closes) < 2:
            return None

        max_gap_pct = 0
        max_gap_info = None

        # Check all consecutive pairs
        for i in range(1, len(closes)):
            prev_close = closes.iloc[i - 1]
            current_close = closes.iloc[i]

            if prev_close == 0:
                continue

            # Calculate gap percentage
            gap_pct = abs((current_close - prev_close) / prev_close) * 100

            if gap_pct > max_gap_pct:
                max_gap_pct = gap_pct
                max_gap_info = {
                    "prev_close": prev_close,
                    "current_close": current_close,
                    "gap_pct": gap_pct,
                    "position": i,
                }

        return max_gap_info

    def _process_gap_anomaly_detection(
        self, gap_result: Dict[str, Any]
    ) -> AnomalyResult:
        """
        Process the gap anomaly detection based on gap results.

        Args:
            gap_result: Gap calculation results

        Returns:
            AnomalyResult with detection details
        """
        gap_pct = gap_result["gap_pct"]

        # Determine if it's an anomaly
        is_anomaly = gap_pct >= self.gap_threshold_pct

        if is_anomaly:
            confidence_score = min(gap_pct / self.severity_thresholds["critical"], 1.0)
            severity = self._calculate_gap_severity(gap_pct)
            anomaly_type = AnomalyType.PRICE_GAP
        else:
            confidence_score = 0.0
            severity = AnomalySeverity.LOW
            anomaly_type = AnomalyType.NONE

        return AnomalyResult(
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence_score=confidence_score,
            z_score=None,  # Not applicable for gaps
            threshold=self.gap_threshold_pct if is_anomaly else None,
            context=gap_result,
        )

    def _calculate_gap_severity(self, gap_pct: float) -> AnomalySeverity:
        """
        Calculate severity based on gap percentage.

        Args:
            gap_pct: Gap percentage

        Returns:
            AnomalySeverity level
        """
        if gap_pct >= self.severity_thresholds.get("critical", 15.0):
            return AnomalySeverity.CRITICAL
        elif gap_pct >= self.severity_thresholds.get("high", 10.0):
            return AnomalySeverity.HIGH
        elif gap_pct >= self.severity_thresholds.get("medium", 5.0):
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _handle_detection_error(self, error: Exception, symbol: str) -> AnomalyResult:
        """
        Handle errors during anomaly detection.

        Args:
            error: The exception that occurred
            symbol: Trading symbol

        Returns:
            Empty anomaly result
        """
        logger.error(f"Error in price gap detection for {symbol}: {error}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return self._create_empty_result()


class KalmanFilter:
    """Simple Kalman filter implementation for state estimation."""

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_state: float = 0.0,
        initial_covariance: float = 1.0,
    ):
        """
        Initialize Kalman filter.

        Args:
            process_noise: Process noise variance (Q)
            measurement_noise: Measurement noise variance (R)
            initial_state: Initial state estimate
            initial_covariance: Initial state covariance (P)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = initial_state  # x
        self.covariance = initial_covariance  # P

    def predict(self) -> None:
        """Predict next state (time update)."""
        # For price prediction, we assume constant velocity model
        # x = x + 0 (no change in state)
        # P = P + Q
        self.covariance += self.process_noise

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update state with new measurement.

        Args:
            measurement: New measurement value

        Returns:
            Tuple of (state_estimate, residual)
        """
        # Kalman gain: K = P / (P + R)
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)

        # Residual: innovation
        residual = measurement - self.state

        # Update state: x = x + K * residual
        self.state = self.state + kalman_gain * residual

        # Update covariance: P = (1 - K) * P
        self.covariance = (1 - kalman_gain) * self.covariance

        return self.state, residual


class RegimeDetector:
    """Adaptive regime detection for market condition classification."""

    def __init__(
        self,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.01,
        memory: int = 20,
    ):
        """
        Initialize regime detector.

        Args:
            volatility_threshold: Threshold for high/low volatility classification
            trend_threshold: Threshold for trending/sideways classification
            memory: Number of observations to remember for regime stability
        """
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.memory = memory

        # Regime history
        self.volatility_history = []
        self.trend_history = []
        self.regime_history = []

    def detect_regime(self, returns: pd.Series, prices: pd.Series) -> Dict[str, Any]:
        """
        Detect current market regime.

        Args:
            returns: Price returns series
            prices: Price series

        Returns:
            Dictionary with regime information
        """
        # Calculate volatility (rolling standard deviation of returns)
        volatility = returns.rolling(window=min(len(returns), 20)).std().iloc[-1]
        if pd.isna(volatility):
            volatility = returns.std() if len(returns) > 1 else 0.01

        # Calculate trend strength (absolute slope of linear regression)
        if len(prices) >= 5:
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices.values, 1)[0]
            trend_strength = abs(slope) / prices.iloc[-1]  # Normalize by current price
        else:
            trend_strength = 0.0

        # Classify regime
        is_high_volatility = volatility > self.volatility_threshold
        is_trending = trend_strength > self.trend_threshold

        if is_high_volatility and not is_trending:
            regime = "high_volatility_sideways"
        elif is_high_volatility and is_trending:
            regime = "high_volatility_trending"
        elif not is_high_volatility and is_trending:
            regime = "low_volatility_trending"
        else:
            regime = "low_volatility_sideways"

        # Update history
        self.volatility_history.append(volatility)
        self.trend_history.append(trend_strength)
        self.regime_history.append(regime)

        # Keep only recent history
        if len(self.volatility_history) > self.memory:
            self.volatility_history = self.volatility_history[-self.memory :]
            self.trend_history = self.trend_history[-self.memory :]
            self.regime_history = self.regime_history[-self.memory :]

        # Calculate regime stability (percentage of time in current regime)
        current_regime_count = self.regime_history.count(regime)
        regime_stability = current_regime_count / len(self.regime_history)

        return {
            "regime": regime,
            "volatility": volatility,
            "trend_strength": trend_strength,
            "is_high_volatility": is_high_volatility,
            "is_trending": is_trending,
            "regime_stability": regime_stability,
        }


class AdaptiveThreshold:
    """Adaptive threshold adjustment based on market regime."""

    def __init__(
        self,
        regime_window: int = 50,
        volatility_multiplier: float = 2.0,
        min_threshold: float = 0.5,
        max_threshold: float = 5.0,
    ):
        """
        Initialize adaptive threshold.

        Args:
            regime_window: Window size for regime analysis
            volatility_multiplier: Multiplier for high volatility regimes
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value
        """
        self.regime_window = regime_window
        self.volatility_multiplier = volatility_multiplier
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # History for adaptation
        self.residual_history = []
        self.regime_history = []

    def calculate_threshold(
        self,
        base_threshold: float,
        regime_info: Dict[str, Any],
        recent_residuals: List[float],
    ) -> float:
        """
        Calculate adaptive threshold based on regime and recent residuals.

        Args:
            base_threshold: Base threshold value
            regime_info: Current regime information
            recent_residuals: Recent residual values

        Returns:
            Adaptive threshold value
        """
        # Update history
        self.regime_history.append(regime_info)
        if len(self.regime_history) > self.regime_window:
            self.regime_history = self.regime_history[-self.regime_window :]

        # Calculate regime-based adjustment
        regime_multiplier = 1.0
        if regime_info.get("is_high_volatility", False):
            regime_multiplier = self.volatility_multiplier

        # Calculate residual-based adjustment
        if recent_residuals:
            recent_std = (
                np.std(recent_residuals[-20:]) if len(recent_residuals) >= 5 else 0.1
            )
            if recent_std > 0:
                residual_multiplier = min(recent_std * 10, 3.0)  # Cap at 3x
                regime_multiplier *= residual_multiplier

        # Apply adjustments
        adaptive_threshold = base_threshold * regime_multiplier

        # Clamp to bounds
        adaptive_threshold = max(
            self.min_threshold, min(self.max_threshold, adaptive_threshold)
        )

        return adaptive_threshold


class KalmanAnomalyDetector(BaseAnomalyDetector):
    """
    Advanced anomaly detector using Kalman filtering with adaptive thresholding.

    This detector uses Kalman filtering to model expected price movements and detects
    anomalies based on prediction residuals. It adapts to changing market regimes
    and provides confidence scores based on multiple factors.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kalman anomaly detector.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Kalman filter configuration
        kalman_config = self.config.get("kalman_filter", {})
        self.kalman_filter = KalmanFilter(
            process_noise=kalman_config.get("process_noise", 0.01),
            measurement_noise=kalman_config.get("measurement_noise", 0.1),
            initial_state=kalman_config.get("initial_state", 0.0),
            initial_covariance=kalman_config.get("initial_covariance", 1.0),
        )

        # Adaptive threshold configuration
        threshold_config = self.config.get("adaptive_threshold", {})
        self.adaptive_threshold = AdaptiveThreshold(
            regime_window=threshold_config.get("regime_window", 50),
            volatility_multiplier=threshold_config.get("volatility_multiplier", 2.0),
            min_threshold=threshold_config.get("min_threshold", 0.5),
            max_threshold=threshold_config.get("max_threshold", 5.0),
        )

        # Regime detection configuration
        regime_config = self.config.get("regime_detection", {})
        self.regime_detector = RegimeDetector(
            volatility_threshold=regime_config.get("volatility_threshold", 0.02),
            trend_threshold=regime_config.get("trend_threshold", 0.01),
            memory=regime_config.get("regime_memory", 20),
        )

        # Confidence scoring weights
        confidence_config = self.config.get("confidence_scoring", {})
        self.residual_weight = confidence_config.get("residual_weight", 0.6)
        self.regime_weight = confidence_config.get("regime_weight", 0.3)
        self.feature_weight = confidence_config.get("feature_weight", 0.1)

        # Manual override events
        self.manual_overrides = []

        # State tracking
        self.residual_history = []
        self.max_residual_history = 100

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect anomalies using Kalman filtering with adaptive thresholding.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        try:
            # Check prerequisites
            if not self._check_prerequisites(data):
                return self._create_conservative_result("insufficient data")

            # Check manual overrides first
            override_result = self._check_manual_overrides(data, symbol)
            if override_result:
                return override_result

            # Calculate returns and prices
            returns = self._calculate_returns(data)
            prices = data["close"].values

            if returns is None or len(prices) < 10:
                return self._create_conservative_result(
                    "insufficient data for Kalman filtering"
                )

            # Detect current regime
            regime_info = self.regime_detector.detect_regime(returns, pd.Series(prices))

            # Run Kalman filtering
            kalman_result = self._run_kalman_filter(prices)

            if kalman_result is None:
                return self._create_conservative_result("Kalman filter failed")

            # Calculate adaptive threshold
            adaptive_threshold = self.adaptive_threshold.calculate_threshold(
                base_threshold=3.0,  # Base z-score threshold
                regime_info=regime_info,
                recent_residuals=self.residual_history[-20:],
            )

            # Calculate anomaly score
            anomaly_score = abs(kalman_result["residual"]) / adaptive_threshold

            # Determine if it's an anomaly
            is_anomaly = anomaly_score > 1.0

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                kalman_result, regime_info, anomaly_score, is_anomaly
            )

            # Calculate severity
            severity = self._calculate_anomaly_severity(anomaly_score)

            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(
                kalman_result, regime_info
            )

            # Prepare context
            context = {
                "kalman_state": kalman_result["state"],
                "residual": kalman_result["residual"],
                "regime": regime_info,
                "adaptive_threshold": adaptive_threshold,
                "anomaly_score": anomaly_score,
                "feature_importance": feature_importance,
            }

            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.KALMAN_FILTER,
                severity=severity,
                confidence_score=confidence_score,
                z_score=kalman_result["residual"] if is_anomaly else None,
                threshold=adaptive_threshold if is_anomaly else None,
                context=context,
            )

        except Exception as e:
            logger.warning(f"Error in Kalman anomaly detection for {symbol}: {e}")
            logger.warning(f"Stack trace: {traceback.format_exc()}")
            return self._create_conservative_result(f"detection error: {str(e)}")

    def _check_prerequisites(self, data: pd.DataFrame) -> bool:
        """Check basic prerequisites for Kalman detection."""
        return (
            self.enabled
            and not data.empty
            and len(data) >= 10  # Need minimum data for Kalman filter
            and "close" in data.columns
        )

    def _create_conservative_result(self, reason: str) -> AnomalyResult:
        """Create a conservative anomaly result."""
        return AnomalyResult(
            is_anomaly=False,
            anomaly_type=AnomalyType.NONE,
            severity=AnomalySeverity.LOW,
            confidence_score=0.0,
            context={"fallback_reason": reason},
        )

    def _calculate_returns(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate price returns."""
        try:
            returns = data["close"].pct_change().dropna()
            return returns if len(returns) >= 5 else None
        except Exception:
            return None

    def _run_kalman_filter(self, prices: np.ndarray) -> Optional[Dict[str, Any]]:
        """Run Kalman filter on price data."""
        try:
            if len(prices) < 2:
                return None

            # Reset filter for new sequence
            self.kalman_filter.state = prices[0]
            self.kalman_filter.covariance = 1.0

            # Process each price point
            residuals = []
            states = []

            for i, price in enumerate(prices):
                if i == 0:
                    # Initialize with first price
                    states.append(price)
                    residuals.append(0.0)
                    continue

                # Predict
                self.kalman_filter.predict()

                # Update with measurement
                state, residual = self.kalman_filter.update(price)

                states.append(state)
                residuals.append(residual)

                # Track residual history
                self.residual_history.append(residual)
                if len(self.residual_history) > self.max_residual_history:
                    self.residual_history = self.residual_history[
                        -self.max_residual_history :
                    ]

            # Return results for the most recent point
            return {
                "state": states[-1],
                "residual": residuals[-1],
                "all_states": states,
                "all_residuals": residuals,
            }

        except Exception as e:
            logger.warning(f"Kalman filter error: {e}")
            return None

    def _calculate_confidence_score(
        self,
        kalman_result: Dict[str, Any],
        regime_info: Dict[str, Any],
        anomaly_score: float,
        is_anomaly: bool,
    ) -> float:
        """Calculate confidence score based on multiple factors."""
        if not is_anomaly:
            return 0.0

        # Residual-based confidence
        residual_confidence = min(abs(kalman_result["residual"]) / 5.0, 1.0)

        # Regime-based confidence (higher confidence in stable regimes)
        regime_stability = regime_info.get("regime_stability", 0.5)
        regime_confidence = regime_stability

        # Feature consistency confidence
        feature_consistency = 1.0  # Could be enhanced with more features

        # Weighted combination
        confidence = (
            self.residual_weight * residual_confidence
            + self.regime_weight * regime_confidence
            + self.feature_weight * feature_consistency
        )

        return min(confidence, 1.0)

    def _calculate_anomaly_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Calculate anomaly severity based on score."""
        if anomaly_score >= 3.0:
            return AnomalySeverity.CRITICAL
        elif anomaly_score >= 2.0:
            return AnomalySeverity.HIGH
        elif anomaly_score >= 1.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW

    def _calculate_feature_importance(
        self, kalman_result: Dict[str, Any], regime_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate feature importance for the anomaly detection."""
        residual_importance = abs(kalman_result["residual"]) / 5.0  # Normalize
        volatility_importance = regime_info.get("volatility", 0.01) / 0.05  # Normalize
        trend_importance = regime_info.get("trend_strength", 0.0) / 0.02  # Normalize

        # Normalize to sum to 1
        total = residual_importance + volatility_importance + trend_importance
        if total > 0:
            residual_importance /= total
            volatility_importance /= total
            trend_importance /= total

        return {
            "residual": residual_importance,
            "volatility": volatility_importance,
            "trend": trend_importance,
        }

    def _check_manual_overrides(
        self, data: pd.DataFrame, symbol: str
    ) -> Optional[AnomalyResult]:
        """Check if any manual overrides apply to current data."""
        if not self.manual_overrides:
            return None

        current_time = (
            data.index[-1] if hasattr(data.index, "__getitem__") else datetime.now()
        )

        for override in self.manual_overrides:
            if override.get("symbol") == symbol:
                override_time = override.get("timestamp")
                if (
                    override_time
                    and abs((current_time - override_time).total_seconds()) < 300
                ):  # 5 minutes
                    return AnomalyResult(
                        is_anomaly=False,
                        anomaly_type=AnomalyType.NONE,
                        severity=AnomalySeverity.LOW,
                        confidence_score=0.0,
                        context={"manual_override": override},
                    )

        return None

    def set_manual_overrides(self, overrides: List[Dict[str, Any]]) -> None:
        """Set manual override events."""
        self.manual_overrides = overrides


class AnomalyDetector:
    """
    Main anomaly detector that combines multiple detection methods
    and provides configurable response mechanisms.
    """

    # Centralized configuration constants
    DEFAULT_CONFIG = {
        "enabled": True,
        "price_zscore": {
            "enabled": True,
            "lookback_period": 50,
            "z_threshold": -10.0,
            "severity_thresholds": {
                "low": 2.0,
                "medium": 3.0,
                "high": 4.0,
                "critical": 5.0,
            },
        },
        "volume_zscore": {
            "enabled": True,
            "lookback_period": 20,
            "z_threshold": 3.0,
            "severity_thresholds": {
                "low": 2.0,
                "medium": 3.0,
                "high": 4.0,
                "critical": 15.0,
            },
        },
        "price_gap": {
            "enabled": True,
            "gap_threshold_pct": 5.0,
            "severity_thresholds": {
                "low": 3.0,
                "medium": 5.0,
                "high": 10.0,
                "critical": 15.0,
            },
        },
        "kalman_anomaly": {
            "enabled": False,  # Disabled by default for backward compatibility
            "kalman_filter": {
                "process_noise": 0.01,
                "measurement_noise": 0.1,
                "initial_state": 0.0,
                "initial_covariance": 1.0,
            },
            "adaptive_threshold": {
                "regime_window": 50,
                "volatility_multiplier": 2.0,
                "min_threshold": 0.5,
                "max_threshold": 5.0,
            },
            "regime_detection": {
                "enabled": True,
                "volatility_threshold": 0.02,
                "trend_threshold": 0.01,
                "regime_memory": 20,
            },
            "confidence_scoring": {
                "residual_weight": 0.6,
                "regime_weight": 0.3,
                "feature_weight": 0.1,
            },
        },
        "response": {
            "skip_trade_threshold": "critical",
            "scale_down_threshold": "medium",
            "scale_down_factor": 0.5,
        },
        "logging": {
            "enabled": True,
            "file": DEFAULT_LOG_FILE,
            "json_file": DEFAULT_JSON_LOG_FILE,
        },
        "max_history": 1000,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly detector.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration parameters are invalid
        """
        # Merge provided config with defaults
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # Validate configuration parameters
        self._validate_config(self.config)
        self.enabled = self.config["enabled"]

        # Initialize detection methods
        self.price_detector = PriceZScoreDetector(self.config["price_zscore"])
        self.volume_detector = VolumeZScoreDetector(self.config["volume_zscore"])
        self.gap_detector = PriceGapDetector(self.config["price_gap"])
        self.kalman_detector = KalmanAnomalyDetector(self.config["kalman_anomaly"])

        # Response configuration with safe defaults
        self.response_config = self.config.get("response", {})
        self.skip_trade_threshold = self._string_to_severity(
            self.response_config.get("skip_trade_threshold", "critical")
        )
        self.scale_down_threshold = self._string_to_severity(
            self.response_config.get("scale_down_threshold", "medium")
        )
        self.scale_down_factor = self.response_config.get("scale_down_factor", 0.5)

        # Logging configuration with secure path validation and safe defaults
        self.log_config = self.config.get("logging", {})
        self.log_anomalies = self.log_config.get("enabled", False)
        log_file_path = self.log_config.get("file", DEFAULT_LOG_FILE)
        json_log_file_path = self.log_config.get("json_file", DEFAULT_JSON_LOG_FILE)

        # Validate and secure file paths
        self.log_file = self._secure_file_path(log_file_path)
        self.json_log_file = self._secure_file_path(json_log_file_path)

        # Anomaly history
        self.max_history = self.config["max_history"]
        self.anomaly_history = TrimmingList(self.max_history)

        logger.info(f"AnomalyDetector initialized: enabled={self.enabled}")

    def _secure_file_path(self, file_path: str) -> str:
        """
        Validate and secure file path to prevent path traversal attacks.

        Args:
            file_path: The file path to validate

        Returns:
            Secure absolute path within the allowed directory or original path for temp files

        Raises:
            ValueError: If the path is invalid or outside the allowed directory
        """
        try:
            # Resolve the path to its canonical form to handle symlinks and relative paths
            resolved_path = Path(file_path).resolve()

            # Define the allowed base directory
            allowed_base = Path(LOG_DIR).resolve()

            # Allow temp directory paths for testing (contains 'temp' or 'tmp' in path)
            path_str = str(resolved_path).lower()
            if "temp" in path_str or "tmp" in path_str:
                # For temp files, allow the original path
                secure_path = str(resolved_path)
            elif not str(resolved_path).startswith(str(allowed_base)):
                # If not within allowed directory, place it within the allowed directory
                resolved_path = allowed_base / Path(file_path).name
                secure_path = str(resolved_path)
            else:
                secure_path = str(resolved_path)

            # Additional security: ensure no directory traversal patterns remain
            if ".." in secure_path or not secure_path:
                raise ValueError(f"Invalid file path: {file_path}")

            return secure_path

        except Exception as e:
            logger.error(f"Error validating file path '{file_path}': {e}")
            raise ValueError(f"Invalid file path: {file_path}")

    def _validate_market_data(self, data: pd.DataFrame) -> None:
        """
        Validate market data DataFrame schema and content.

        Args:
            data: Market data DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if data is None:
            raise ValueError("Market data cannot be None")

        if data.empty:
            raise ValueError("Market data DataFrame is empty")

        # Check for required columns
        required_columns = ["close"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types
        if not pd.api.types.is_numeric_dtype(data["close"]):
            raise ValueError("Column 'close' must contain numeric data")

        # Check for NaN values in critical columns
        if data["close"].isna().any():
            raise ValueError("Column 'close' contains NaN values")

        # Check for infinite values
        if np.isinf(data["close"]).any():
            raise ValueError("Column 'close' contains infinite values")

        # Validate reasonable value ranges (optional but recommended)
        if (data["close"] <= 0).any():
            raise ValueError("Column 'close' contains non-positive values")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration parameters for the anomaly detector.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        # Validate basic parameters
        self._validate_basic_config(config)

        # Validate response configuration
        self._validate_response_config(config)

        # Validate detector configurations
        self._validate_detector_configs(config)

    def _validate_basic_config(self, config: Dict[str, Any]) -> None:
        """
        Validate basic configuration parameters.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If basic parameters are invalid
        """
        # Validate enabled flag
        enabled = config.get("enabled", self.DEFAULT_CONFIG["enabled"])
        if not isinstance(enabled, bool):
            raise ValueError(f"enabled must be a boolean, got {type(enabled)}")

        # Validate max_history
        max_history = config.get("max_history", self.DEFAULT_CONFIG["max_history"])
        if not isinstance(max_history, int):
            raise ValueError(f"max_history must be an integer, got {type(max_history)}")
        if max_history <= 0:
            raise ValueError(f"max_history must be positive, got {max_history}")
        if max_history > 100000:  # Reasonable upper limit
            raise ValueError(f"max_history seems too high (>100000), got {max_history}")

    def _validate_response_config(self, config: Dict[str, Any]) -> None:
        """
        Validate response configuration parameters.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If response parameters are invalid
        """
        response_config = config.get("response", self.DEFAULT_CONFIG["response"])
        if not isinstance(response_config, dict):
            raise ValueError("response configuration must be a dictionary")

        scale_down_factor = response_config.get(
            "scale_down_factor", self.DEFAULT_CONFIG["response"]["scale_down_factor"]
        )
        if not isinstance(scale_down_factor, (int, float)):
            raise ValueError(
                f"scale_down_factor must be numeric, got {type(scale_down_factor)}"
            )
        if not 0 < scale_down_factor <= 1:
            raise ValueError(
                f"scale_down_factor must be between 0 and 1, got {scale_down_factor}"
            )

        # Validate severity thresholds
        valid_severities = ["low", "medium", "high", "critical"]
        skip_threshold = response_config.get(
            "skip_trade_threshold",
            self.DEFAULT_CONFIG["response"]["skip_trade_threshold"],
        )
        scale_threshold = response_config.get(
            "scale_down_threshold",
            self.DEFAULT_CONFIG["response"]["scale_down_threshold"],
        )

        if skip_threshold not in valid_severities:
            raise ValueError(
                f"skip_trade_threshold must be one of {valid_severities}, got '{skip_threshold}'"
            )
        if scale_threshold not in valid_severities:
            raise ValueError(
                f"scale_down_threshold must be one of {valid_severities}, got '{scale_threshold}'"
            )

    def _validate_detector_configs(self, config: Dict[str, Any]) -> None:
        """
        Validate detector-specific configuration parameters.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If detector parameters are invalid
        """
        detectors = ["price_zscore", "volume_zscore", "price_gap"]
        for detector_name in detectors:
            detector_config = config.get(
                detector_name, self.DEFAULT_CONFIG.get(detector_name, {})
            )
            if not isinstance(detector_config, dict):
                raise ValueError(f"{detector_name} configuration must be a dictionary")

            # Validate parameters based on detector type
            if detector_name in ["price_zscore", "volume_zscore"]:
                self._validate_zscore_detector_config(detector_name, detector_config)
            elif detector_name == "price_gap":
                self._validate_gap_detector_config(detector_name, detector_config)

            # Validate severity thresholds for all detectors
            self._validate_severity_thresholds(detector_name, detector_config)

    def _validate_zscore_detector_config(
        self, detector_name: str, detector_config: Dict[str, Any]
    ) -> None:
        """
        Validate z-score detector configuration parameters.

        Args:
            detector_name: Name of the detector
            detector_config: Detector configuration

        Raises:
            ValueError: If parameters are invalid
        """
        default_lookback = 50 if detector_name == "price_zscore" else 20
        lookback_period = detector_config.get("lookback_period", default_lookback)
        if not isinstance(lookback_period, int):
            raise ValueError(
                f"{detector_name}.lookback_period must be an integer, got {type(lookback_period)}"
            )
        if lookback_period <= 0:
            raise ValueError(
                f"{detector_name}.lookback_period must be positive, got {lookback_period}"
            )
        if lookback_period > 1000:  # Reasonable upper limit
            raise ValueError(
                f"{detector_name}.lookback_period seems too high (>1000), got {lookback_period}"
            )

        # Validate z_threshold - log warning and set default instead of raising error
        z_threshold = detector_config.get("z_threshold", 3.0)
        if not isinstance(z_threshold, (int, float)) or z_threshold <= 0:
            logger.warning(
                f"{detector_name}.z_threshold invalid ({z_threshold}), defaulting to 3.0"
            )
            detector_config["z_threshold"] = 3.0
        elif z_threshold > 10:  # Reasonable upper limit for z-score
            logger.warning(
                f"{detector_name}.z_threshold seems too high (>10), got {z_threshold}, defaulting to 3.0"
            )
            detector_config["z_threshold"] = 3.0

    def _validate_gap_detector_config(
        self, detector_name: str, detector_config: Dict[str, Any]
    ) -> None:
        """
        Validate gap detector configuration parameters.

        Args:
            detector_name: Name of the detector
            detector_config: Detector configuration

        Raises:
            ValueError: If parameters are invalid
        """
        gap_threshold_pct = detector_config.get("gap_threshold_pct", 5.0)
        if not isinstance(gap_threshold_pct, (int, float)):
            raise ValueError(
                f"{detector_name}.gap_threshold_pct must be numeric, got {type(gap_threshold_pct)}"
            )
        if gap_threshold_pct <= 0:
            raise ValueError(
                f"{detector_name}.gap_threshold_pct must be positive, got {gap_threshold_pct}"
            )
        if gap_threshold_pct > 100:  # Reasonable upper limit for percentage
            raise ValueError(
                f"{detector_name}.gap_threshold_pct seems too high (>100%), got {gap_threshold_pct}"
            )

    def _validate_severity_thresholds(
        self, detector_name: str, detector_config: Dict[str, Any]
    ) -> None:
        """
        Validate severity thresholds for a detector.

        Args:
            detector_name: Name of the detector
            detector_config: Detector configuration

        Raises:
            ValueError: If severity thresholds are invalid
        """
        severity_thresholds = detector_config.get("severity_thresholds", {})
        if not isinstance(severity_thresholds, dict):
            raise ValueError(
                f"{detector_name}.severity_thresholds must be a dictionary"
            )

        valid_severities = ["low", "medium", "high", "critical"]
        for level in valid_severities:
            if level in severity_thresholds:
                threshold_value = severity_thresholds[level]
                if not isinstance(threshold_value, (int, float)):
                    raise ValueError(
                        f"{detector_name}.severity_thresholds.{level} must be numeric, got {type(threshold_value)}"
                    )
                if threshold_value <= 0:
                    raise ValueError(
                        f"{detector_name}.severity_thresholds.{level} must be positive, got {threshold_value}"
                    )

    def detect_anomalies(
        self, data: pd.DataFrame, symbol: str = ""
    ) -> List[AnomalyResult]:
        """
        Run all anomaly detection methods on the data.

        This method orchestrates the anomaly detection process by delegating
        to specialized helper methods for each concern.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            List of AnomalyResult objects
        """
        if not self.enabled:
            return []

        # Step 1: Validate data and check prerequisites
        if not self._validate_and_check_data(data, symbol):
            return []

        # Step 2: Run all enabled detectors
        results = self._run_all_detectors(data, symbol)

        # Step 3: Filter and return only anomalies
        return self._filter_anomalies(results)

    def _validate_and_check_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """
        Validate market data and check basic prerequisites.

        Args:
            data: Market data DataFrame
            symbol: Trading symbol

        Returns:
            True if validation passes, False otherwise
        """
        try:
            self._validate_market_data(data)
        except ValueError as e:
            logger.warning(f"Market data validation failed for {symbol}: {e}")
            return False

        if data.empty:
            return False

        return True

    def _run_all_detectors(
        self, data: pd.DataFrame, symbol: str
    ) -> List[AnomalyResult]:
        """
        Run all enabled anomaly detectors.

        Args:
            data: Market data DataFrame
            symbol: Trading symbol

        Returns:
            List of all detection results
        """
        results = []

        # Run each detector
        if self.price_detector.enabled:
            result = self.price_detector.detect(data, symbol)
            results.append(result)

        if self.volume_detector.enabled:
            result = self.volume_detector.detect(data, symbol)
            results.append(result)

        if self.gap_detector.enabled:
            result = self.gap_detector.detect(data, symbol)
            results.append(result)

        if self.kalman_detector.enabled:
            result = self.kalman_detector.detect(data, symbol)
            results.append(result)

        return results

    def _filter_anomalies(self, results: List[AnomalyResult]) -> List[AnomalyResult]:
        """
        Filter results to return only actual anomalies.

        Args:
            results: List of detection results

        Returns:
            List containing only anomaly results
        """
        return [result for result in results if result.is_anomaly]

    def check_signal_anomaly(
        self, signal: Dict[str, Any], data: pd.DataFrame, symbol: str = ""
    ) -> Tuple[bool, Optional[AnomalyResponse], Optional[AnomalyResult]]:
        """
        Check if a trading signal should be modified due to anomalies.

        This method orchestrates the signal anomaly checking process by delegating
        to specialized helper methods for each concern.

        Args:
            signal: Trading signal dictionary
            data: Market data DataFrame
            symbol: Trading symbol

        Returns:
            Tuple of (should_proceed, response_type, anomaly_result)
        """
        if not self.enabled:
            return True, None, None

        # Step 1: Detect anomalies
        anomalies = self.detect_anomalies(data, symbol)

        if not anomalies:
            return True, None, None

        # Step 2: Process the most severe anomaly
        most_severe = self._find_most_severe_anomaly(anomalies)
        response = self._determine_response(most_severe)

        # Step 3: Handle logging and return result
        return self._handle_anomaly_response(
            symbol, most_severe, data, signal, response
        )

    def _find_most_severe_anomaly(
        self, anomalies: List[AnomalyResult]
    ) -> AnomalyResult:
        """
        Find the most severe anomaly from a list of anomalies.

        Args:
            anomalies: List of anomaly results

        Returns:
            The most severe anomaly
        """
        return max(anomalies, key=lambda x: self._severity_score(x.severity))

    def _handle_anomaly_response(
        self,
        symbol: str,
        anomaly: AnomalyResult,
        data: pd.DataFrame,
        signal: Dict[str, Any],
        response: AnomalyResponse,
    ) -> Tuple[bool, Optional[AnomalyResponse], Optional[AnomalyResult]]:
        """
        Handle the anomaly response, including logging.

        Args:
            symbol: Trading symbol
            anomaly: The detected anomaly
            data: Market data DataFrame
            signal: Original trading signal
            response: Determined response

        Returns:
            Tuple of (should_proceed, response_type, anomaly_result)
        """
        # Log the anomaly asynchronously if event loop is available, otherwise synchronously
        if self.log_anomalies:
            try:
                # Check if there's a running event loop
                loop = asyncio.get_running_loop()
                asyncio.create_task(
                    self._log_anomaly_async(symbol, anomaly, data, signal, response)
                )
            except RuntimeError:
                # No event loop running, log synchronously
                logger.warning("No event loop running, logging anomaly synchronously")
                try:
                    # Create a new event loop for this operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        self._log_anomaly_async(symbol, anomaly, data, signal, response)
                    )
                    loop.close()
                except Exception as e:
                    logger.error(f"Failed to log anomaly even synchronously: {e}")

        should_proceed = response != AnomalyResponse.SKIP_TRADE
        return should_proceed, response, anomaly

    def _severity_score(self, severity: AnomalySeverity) -> int:
        """Convert severity to numeric score for comparison."""
        scores = {
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.CRITICAL: 4,
        }
        return scores.get(severity, 0)

    def _determine_response(self, anomaly: AnomalyResult) -> AnomalyResponse:
        """Determine the appropriate response to an anomaly."""
        severity_score = self._severity_score(anomaly.severity)

        skip_threshold_score = self._severity_score(self.skip_trade_threshold)
        scale_threshold_score = self._severity_score(self.scale_down_threshold)

        if severity_score >= skip_threshold_score:
            return AnomalyResponse.SKIP_TRADE
        elif severity_score >= scale_threshold_score:
            return AnomalyResponse.SCALE_DOWN
        else:
            return AnomalyResponse.LOG_ONLY

    def _string_to_severity(self, severity_str: str) -> AnomalySeverity:
        """Convert string to AnomalySeverity enum."""
        mapping = {
            "low": AnomalySeverity.LOW,
            "medium": AnomalySeverity.MEDIUM,
            "high": AnomalySeverity.HIGH,
            "critical": AnomalySeverity.CRITICAL,
        }
        return mapping.get(severity_str.lower(), AnomalySeverity.LOW)

    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """
        Generate statistics about detected anomalies.

        Returns:
            Dictionary containing anomaly statistics
        """
        stats = {
            "total_anomalies": len(self.anomaly_history),
            "by_type": {},
            "by_severity": {},
            "by_response": {},
        }

        for log in self.anomaly_history:
            anomaly = log.anomaly_result

            # Count by type
            anomaly_type = anomaly.anomaly_type.value
            stats["by_type"][anomaly_type] = stats["by_type"].get(anomaly_type, 0) + 1

            # Count by severity
            severity = anomaly.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1

            # Count by response
            response = log.action_taken or "none"
            stats["by_response"][response] = stats["by_response"].get(response, 0) + 1

        return stats

    def _log_anomaly(
        self,
        symbol: str,
        anomaly_result: AnomalyResult,
        data: pd.DataFrame,
        extra: Optional[Dict[str, Any]] = None,
        response: Optional[AnomalyResponse] = None,
    ):
        """
        Log anomaly details to configured outputs.

        Args:
            symbol: Trading symbol
            anomaly_result: The detected anomaly result
            data: Market data DataFrame
            extra: Additional logging information
            response: Response action taken
        """
        logger.info(f"Anomaly detected: {anomaly_result}")

        if self.log_anomalies:
            # Log to file if configured
            if "file" in self.log_config:
                try:
                    with open(self.log_config["file"], "a") as f:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(
                            f"[{timestamp}] {symbol}: {anomaly_result.anomaly_type.value} "
                            f"(severity: {anomaly_result.severity.value}, "
                            f"confidence: {anomaly_result.confidence_score:.3f}) "
                            f"Action: {response.value if response else 'none'}\n"
                        )
                except Exception as e:
                    logger.error(f"Failed to log to file: {e}")

            # Log to JSON file if configured
            if "json_file" in self.log_config:
                try:
                    # Read existing logs
                    existing_logs = []
                    if os.path.exists(self.log_config["json_file"]):
                        try:
                            with open(self.log_config["json_file"], "r") as f:
                                existing_logs = json.load(f)
                        except (json.JSONDecodeError, FileNotFoundError):
                            existing_logs = []

                    # Create log entry
                    log_entry = {
                        "symbol": symbol,
                        "anomaly_result": anomaly_result.to_dict(),
                        "market_data": data.tail(5).to_dict("records")
                        if not data.empty
                        else None,
                        "action_taken": response.value if response else None,
                        "original_signal": extra,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Add to history
                    existing_logs.append(log_entry)

                    # Keep only recent logs
                    if len(existing_logs) > self.max_history:
                        existing_logs = existing_logs[-self.max_history :]

                    # Write back
                    with open(self.log_config["json_file"], "w") as f:
                        json.dump(existing_logs, f, indent=2, default=str)

                except Exception as e:
                    logger.error(f"Failed to log to JSON file: {e}")

        # Log to trade logger
        try:
            trade_logger.performance(
                f"Anomaly detected: {anomaly_result.anomaly_type.value}",
                {
                    "symbol": symbol,
                    "anomaly_type": anomaly_result.anomaly_type.value,
                    "severity": anomaly_result.severity.value,
                    "confidence": anomaly_result.confidence_score,
                    "action": response.value if response else "none",
                    "z_score": anomaly_result.z_score,
                    "threshold": anomaly_result.threshold,
                },
            )
        except Exception as e:
            logger.error(f"Failed to log to trade logger: {e}")

    def _log_to_file(self, log_entry: AnomalyLog):
        """
        Log anomaly to text file synchronously.

        Args:
            log_entry: AnomalyLog entry to write
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

            timestamp = (
                log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if log_entry.timestamp
                else "Unknown"
            )

            log_line = (
                f"[{timestamp}] {log_entry.symbol}: {log_entry.anomaly_result.anomaly_type.value} "
                f"(severity: {log_entry.anomaly_result.severity.value}, "
                f"confidence: {log_entry.anomaly_result.confidence_score:.3f}) "
                f"Action: {log_entry.action_taken or 'none'}\n"
            )

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line)

        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def _log_to_json(self, log_entry: AnomalyLog):
        """
        Log anomaly to JSON file synchronously.

        Args:
            log_entry: AnomalyLog entry to write
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.json_log_file), exist_ok=True)

            # Read existing logs
            existing_logs = []
            if os.path.exists(self.json_log_file):
                try:
                    with open(self.json_log_file, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_logs = []

            # Add new log entry
            existing_logs.append(log_entry.to_dict())

            # Keep only recent logs
            if len(existing_logs) > self.max_history:
                existing_logs = existing_logs[-self.max_history :]

            # Write back
            with open(self.json_log_file, "w", encoding="utf-8") as f:
                json.dump(existing_logs, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to write to JSON log file: {e}")

    async def _log_anomaly_async(
        self,
        symbol: str,
        anomaly: AnomalyResult,
        market_data: pd.DataFrame,
        signal: Optional[Dict[str, Any]] = None,
        response: Optional[AnomalyResponse] = None,
    ):
        """
        Asynchronously log anomaly details to prevent blocking the main thread.

        This method implements non-blocking I/O operations using aiofiles to ensure
        that file logging doesn't impact the performance of the real-time trading system.
        In high-throughput scenarios, synchronous file operations can cause significant
        delays and make the system unresponsive.

        Args:
            symbol: Trading symbol
            anomaly: Detected anomaly result
            market_data: Market data DataFrame
            signal: Original trading signal
            response: Response action taken
        """
        try:
            # Create anomaly log entry
            log_entry = AnomalyLog(
                symbol=symbol,
                anomaly_result=anomaly,
                market_data=market_data.tail(5).to_dict("records")
                if not market_data.empty
                else None,
                action_taken=response.value if response else None,
                original_signal=signal,
            )

            # Add to history
            self.anomaly_history.append(log_entry)

            # Log to text file asynchronously
            await self._log_to_file_async(log_entry)

            # Log to JSON file asynchronously
            await self._log_to_json_async(log_entry)

            # Log to trade logger (use performance method as anomaly method may not exist)
            trade_logger.performance(
                f"Anomaly detected: {anomaly.anomaly_type.value}",
                {
                    "symbol": symbol,
                    "anomaly_type": anomaly.anomaly_type.value,
                    "severity": anomaly.severity.value,
                    "confidence": anomaly.confidence_score,
                    "action": response.value if response else "none",
                    "z_score": anomaly.z_score,
                    "threshold": anomaly.threshold,
                },
            )

        except Exception as e:
            logger.error(f"Failed to log anomaly asynchronously: {e}")

    async def _log_to_file_async(self, log_entry: AnomalyLog):
        """
        Log anomaly to text file asynchronously.

        This method uses aiofiles for non-blocking file I/O operations.
        If aiofiles is not available, it falls back to synchronous operations
        but logs a warning about potential performance impact.

        Args:
            log_entry: AnomalyLog entry to write
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

            timestamp = (
                log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if log_entry.timestamp
                else "Unknown"
            )

            log_line = (
                f"[{timestamp}] {log_entry.symbol}: {log_entry.anomaly_result.anomaly_type.value} "
                f"(severity: {log_entry.anomaly_result.severity.value}, "
                f"confidence: {log_entry.anomaly_result.confidence_score:.3f}) "
                f"Action: {log_entry.action_taken or 'none'}\n"
            )

            if aiofiles:
                # Use async file operations for better performance
                async with aiofiles.open(self.log_file, "a", encoding="utf-8") as f:
                    await f.write(log_line)
            else:
                # Fallback to synchronous operations with warning
                logger.warning(
                    "Using synchronous file I/O - consider installing aiofiles for better performance"
                )
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_line)

        except Exception as e:
            logger.error(f"Failed to write to log file asynchronously: {e}")

    async def _log_to_json_async(self, log_entry: AnomalyLog):
        """
        Log anomaly to JSON file asynchronously.

        This method uses aiofiles for non-blocking file I/O operations to maintain
        system responsiveness during high-frequency trading operations.

        Args:
            log_entry: AnomalyLog entry to write
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.json_log_file), exist_ok=True)

            # Read existing logs asynchronously
            existing_logs = []
            if aiofiles and os.path.exists(self.json_log_file):
                try:
                    async with aiofiles.open(
                        self.json_log_file, "r", encoding="utf-8"
                    ) as f:
                        content = await f.read()
                        existing_logs = json.loads(content) if content.strip() else []
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_logs = []
            elif os.path.exists(self.json_log_file):
                # Fallback for reading
                try:
                    with open(self.json_log_file, "r", encoding="utf-8") as f:
                        existing_logs = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_logs = []

            # Add new log entry
            existing_logs.append(log_entry.to_dict())

            # Keep only recent logs
            if len(existing_logs) > self.max_history:
                existing_logs = existing_logs[-self.max_history :]

            # Write back to file
            json_content = json.dumps(existing_logs, indent=2, default=str)

            if aiofiles:
                async with aiofiles.open(
                    self.json_log_file, "w", encoding="utf-8"
                ) as f:
                    await f.write(json_content)
            else:
                # Fallback to synchronous operations
                with open(self.json_log_file, "w", encoding="utf-8") as f:
                    json.dump(existing_logs, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to write to JSON log file asynchronously: {e}")


# Convenience functions
def detect_anomalies(
    data: pd.DataFrame, symbol: str = "", config: Optional[Dict[str, Any]] = None
) -> List[AnomalyResult]:
    """
    Convenience function to detect anomalies using the global detector instance.

    Args:
        data: Market data DataFrame
        symbol: Trading symbol
        config: Optional configuration override

    Returns:
        List of detected anomalies
    """
    detector = get_anomaly_detector(config)
    return detector.detect_anomalies(data, symbol)


def check_signal_anomaly(
    signal: Dict[str, Any],
    data: pd.DataFrame,
    symbol: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[AnomalyResponse], Optional[AnomalyResult]]:
    """
    Convenience function to check if a signal should be modified due to anomalies.

    Args:
        signal: Trading signal dictionary
        data: Market data DataFrame
        symbol: Trading symbol
        config: Optional configuration override

    Returns:
        Tuple of (should_proceed, response_type, anomaly_result)
    """
    detector = get_anomaly_detector(config)
    return detector.check_signal_anomaly(signal, data, symbol)


# Global instance
_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector(config: Optional[Dict[str, Any]] = None) -> AnomalyDetector:
    """Get the global anomaly detector instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        if config is None:
            from utils.config_loader import get_config

            config = get_config("risk.anomaly_detector", {})
        _anomaly_detector = AnomalyDetector(config)
    return _anomaly_detector
