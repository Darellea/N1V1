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
"""

import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from statistics import mean, stdev

from utils.config_loader import get_config
from utils.logger import get_trade_logger

logger = logging.getLogger(__name__)
trade_logger = get_trade_logger()


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    PRICE_ZSCORE = "price_zscore"
    VOLUME_ZSCORE = "volume_zscore"
    PRICE_GAP = "price_gap"
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
        data['anomaly_type'] = self.anomaly_type.value
        data['severity'] = self.severity.value
        data['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
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
            'symbol': self.symbol,
            'anomaly_result': self.anomaly_result.to_dict(),
            'market_data': self.market_data,
            'action_taken': self.action_taken,
            'original_signal': self.original_signal,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class BaseAnomalyDetector:
    """Base class for anomaly detection methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
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

    def _calculate_severity(self, z_score: float, thresholds: Dict[str, float]) -> AnomalySeverity:
        """Calculate severity based on z-score and thresholds."""
        abs_z = abs(z_score)

        if abs_z >= thresholds.get('critical', 5.0):
            return AnomalySeverity.CRITICAL
        elif abs_z >= thresholds.get('high', 3.0):
            return AnomalySeverity.HIGH
        elif abs_z >= thresholds.get('medium', 2.0):
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class PriceZScoreDetector(BaseAnomalyDetector):
    """Detect price anomalies using statistical z-score analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lookback_period = self.config.get('lookback_period', 50)
        self.z_threshold = self.config.get('z_threshold', 3.0)
        self.severity_thresholds = self.config.get('severity_thresholds', {
            'low': 2.0,
            'medium': 3.0,
            'high': 4.0,
            'critical': 5.0
        })

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect price anomalies using z-score analysis.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        if not self.enabled or data.empty or len(data) < self.lookback_period:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                severity=AnomalySeverity.LOW,
                confidence_score=0.0
            )

        try:
            # Calculate returns
            if 'close' not in data.columns:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            returns = data['close'].pct_change().dropna()

            if len(returns) < self.lookback_period:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Calculate z-score for the most recent return
            recent_returns = returns.tail(self.lookback_period)
            current_return = recent_returns.iloc[-1]

            if len(recent_returns) < 2:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Calculate mean and std of historical returns
            historical_returns = recent_returns.iloc[:-1]  # Exclude current return
            mean_return = historical_returns.mean()
            std_return = historical_returns.std()

            if std_return == 0 or np.isnan(std_return):
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Calculate z-score
            z_score = (current_return - mean_return) / std_return
            abs_z_score = abs(z_score)

            # Determine if it's an anomaly
            is_anomaly = abs_z_score >= self.z_threshold

            # Calculate confidence score (0-1)
            confidence_score = min(abs_z_score / self.severity_thresholds['critical'], 1.0)

            # Determine severity
            severity = self._calculate_severity(z_score, self.severity_thresholds)

            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.PRICE_ZSCORE,
                severity=severity,
                confidence_score=confidence_score,
                z_score=float(z_score),
                threshold=self.z_threshold,
                context={
                    'current_return': float(current_return),
                    'mean_return': float(mean_return),
                    'std_return': float(std_return),
                    'lookback_period': self.lookback_period
                }
            )

        except Exception as e:
            logger.warning(f"Error in price z-score detection: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                severity=AnomalySeverity.LOW,
                confidence_score=0.0
            )


class VolumeZScoreDetector(BaseAnomalyDetector):
    """Detect volume anomalies using statistical z-score analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.lookback_period = self.config.get('lookback_period', 20)
        self.z_threshold = self.config.get('z_threshold', 3.0)
        self.severity_thresholds = self.config.get('severity_thresholds', {
            'low': 2.0,
            'medium': 3.0,
            'high': 4.0,
            'critical': 5.0
        })

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect volume anomalies using z-score analysis.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        if not self.enabled or data.empty or len(data) < self.lookback_period:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                severity=AnomalySeverity.LOW,
                confidence_score=0.0
            )

        try:
            if 'volume' not in data.columns:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            volumes = data['volume'].dropna()

            if len(volumes) < self.lookback_period:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Get current volume
            current_volume = volumes.iloc[-1]

            # Calculate z-score for the current volume
            recent_volumes = volumes.tail(self.lookback_period)

            if len(recent_volumes) < 2:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Calculate mean and std of historical volumes
            historical_volumes = recent_volumes.iloc[:-1]  # Exclude current volume
            mean_volume = historical_volumes.mean()
            std_volume = historical_volumes.std()

            if std_volume == 0 or np.isnan(std_volume) or mean_volume == 0:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Calculate z-score
            z_score = (current_volume - mean_volume) / std_volume
            abs_z_score = abs(z_score)

            # Determine if it's an anomaly
            is_anomaly = abs_z_score >= self.z_threshold

            # Calculate confidence score (0-1)
            confidence_score = min(abs_z_score / self.severity_thresholds['critical'], 1.0)

            # Determine severity
            severity = self._calculate_severity(z_score, self.severity_thresholds)

            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.VOLUME_ZSCORE,
                severity=severity,
                confidence_score=confidence_score,
                z_score=float(z_score),
                threshold=self.z_threshold,
                context={
                    'current_volume': float(current_volume),
                    'mean_volume': float(mean_volume),
                    'std_volume': float(std_volume),
                    'lookback_period': self.lookback_period
                }
            )

        except Exception as e:
            logger.warning(f"Error in volume z-score detection: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                severity=AnomalySeverity.LOW,
                confidence_score=0.0
            )


class PriceGapDetector(BaseAnomalyDetector):
    """Detect price gap anomalies (large jumps between consecutive candles)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.gap_threshold_pct = self.config.get('gap_threshold_pct', 5.0)  # 5% gap
        self.severity_thresholds = self.config.get('severity_thresholds', {
            'low': 3.0,      # 3% gap
            'medium': 5.0,   # 5% gap
            'high': 10.0,    # 10% gap
            'critical': 15.0 # 15% gap
        })

    def detect(self, data: pd.DataFrame, symbol: str = "") -> AnomalyResult:
        """
        Detect price gap anomalies.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            AnomalyResult with detection details
        """
        if not self.enabled or data.empty or len(data) < 2:
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                severity=AnomalySeverity.LOW,
                confidence_score=0.0
            )

        try:
            if 'close' not in data.columns:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Calculate gap between consecutive closes
            closes = data['close']
            prev_close = closes.iloc[-2]
            current_close = closes.iloc[-1]

            if prev_close == 0:
                return AnomalyResult(
                    is_anomaly=False,
                    anomaly_type=AnomalyType.NONE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.0
                )

            # Calculate gap percentage
            gap_pct = abs((current_close - prev_close) / prev_close) * 100

            # Determine if it's an anomaly
            is_anomaly = gap_pct >= self.gap_threshold_pct

            # Calculate confidence score (0-1)
            confidence_score = min(gap_pct / self.severity_thresholds['critical'], 1.0)

            # Determine severity
            severity = self._calculate_gap_severity(gap_pct)

            return AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_type=AnomalyType.PRICE_GAP,
                severity=severity,
                confidence_score=confidence_score,
                z_score=None,  # Not applicable for gaps
                threshold=self.gap_threshold_pct,
                context={
                    'prev_close': float(prev_close),
                    'current_close': float(current_close),
                    'gap_pct': float(gap_pct),
                    'gap_threshold': self.gap_threshold_pct
                }
            )

        except Exception as e:
            logger.warning(f"Error in price gap detection: {e}")
            return AnomalyResult(
                is_anomaly=False,
                anomaly_type=AnomalyType.NONE,
                severity=AnomalySeverity.LOW,
                confidence_score=0.0
            )

    def _calculate_gap_severity(self, gap_pct: float) -> AnomalySeverity:
        """Calculate severity based on gap percentage."""
        if gap_pct >= self.severity_thresholds.get('critical', 15.0):
            return AnomalySeverity.CRITICAL
        elif gap_pct >= self.severity_thresholds.get('high', 10.0):
            return AnomalySeverity.HIGH
        elif gap_pct >= self.severity_thresholds.get('medium', 5.0):
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class AnomalyDetector:
    """
    Main anomaly detector that combines multiple detection methods
    and provides configurable response mechanisms.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the anomaly detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.enabled = self.config.get('enabled', True)

        # Initialize detection methods
        self.price_detector = PriceZScoreDetector(self.config.get('price_zscore', {}))
        self.volume_detector = VolumeZScoreDetector(self.config.get('volume_zscore', {}))
        self.gap_detector = PriceGapDetector(self.config.get('price_gap', {}))

        # Response configuration
        self.response_config = self.config.get('response', {})
        self.skip_trade_threshold = self.response_config.get('skip_trade_threshold', AnomalySeverity.HIGH)
        self.scale_down_threshold = self.response_config.get('scale_down_threshold', AnomalySeverity.MEDIUM)
        self.scale_down_factor = self.response_config.get('scale_down_factor', 0.5)

        # Logging configuration
        self.log_anomalies = self.config.get('logging', {}).get('enabled', True)
        self.log_file = self.config.get('logging', {}).get('file', 'logs/anomalies.log')
        self.json_log_file = self.config.get('logging', {}).get('json_file', 'logs/anomalies.json')

        # Anomaly history
        self.anomaly_history: List[AnomalyLog] = []
        self.max_history = self.config.get('max_history', 1000)

        logger.info(f"AnomalyDetector initialized: enabled={self.enabled}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'enabled': True,
            'price_zscore': {
                'enabled': True,
                'lookback_period': 50,
                'z_threshold': 3.0,
                'severity_thresholds': {
                    'low': 2.0,
                    'medium': 3.0,
                    'high': 4.0,
                    'critical': 5.0
                }
            },
            'volume_zscore': {
                'enabled': True,
                'lookback_period': 20,
                'z_threshold': 3.0,
                'severity_thresholds': {
                    'low': 2.0,
                    'medium': 3.0,
                    'high': 4.0,
                    'critical': 5.0
                }
            },
            'price_gap': {
                'enabled': True,
                'gap_threshold_pct': 5.0,
                'severity_thresholds': {
                    'low': 3.0,
                    'medium': 5.0,
                    'high': 10.0,
                    'critical': 15.0
                }
            },
            'response': {
                'skip_trade_threshold': 'high',
                'scale_down_threshold': 'medium',
                'scale_down_factor': 0.5
            },
            'logging': {
                'enabled': True,
                'file': 'logs/anomalies.log',
                'json_file': 'logs/anomalies.json'
            },
            'max_history': 1000
        }

    def detect_anomalies(self, data: pd.DataFrame, symbol: str = "") -> List[AnomalyResult]:
        """
        Run all anomaly detection methods on the data.

        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol

        Returns:
            List of AnomalyResult objects
        """
        if not self.enabled or data.empty:
            return []

        results = []

        # Run each detector
        if self.price_detector.enabled:
            result = self.price_detector.detect(data, symbol)
            if result.is_anomaly:
                results.append(result)

        if self.volume_detector.enabled:
            result = self.volume_detector.detect(data, symbol)
            if result.is_anomaly:
                results.append(result)

        if self.gap_detector.enabled:
            result = self.gap_detector.detect(data, symbol)
            if result.is_anomaly:
                results.append(result)

        return results

    def check_signal_anomaly(self, signal: Dict[str, Any], data: pd.DataFrame,
                           symbol: str = "") -> Tuple[bool, Optional[AnomalyResponse], Optional[AnomalyResult]]:
        """
        Check if a trading signal should be modified due to anomalies.

        Args:
            signal: Trading signal dictionary
            data: Market data DataFrame
            symbol: Trading symbol

        Returns:
            Tuple of (should_proceed, response_type, anomaly_result)
        """
        if not self.enabled:
            return True, None, None

        # Detect anomalies
        anomalies = self.detect_anomalies(data, symbol)

        if not anomalies:
            return True, None, None

        # Find the most severe anomaly
        most_severe = max(anomalies, key=lambda x: self._severity_score(x.severity))

        # Determine response based on severity
        response = self._determine_response(most_severe)

        # Log the anomaly
        if self.log_anomalies:
            self._log_anomaly(symbol, most_severe, data, signal, response)

        return response != AnomalyResponse.SKIP_TRADE, response, most_severe

    def _severity_score(self, severity: AnomalySeverity) -> int:
        """Convert severity to numeric score for comparison."""
        scores = {
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.CRITICAL: 4
        }
        return scores.get(severity, 0)

    def _determine_response(self, anomaly: AnomalyResult) -> AnomalyResponse:
        """Determine the appropriate response to an anomaly."""
        severity_score = self._severity_score(anomaly.severity)

        skip_threshold_score = self._severity_score(self._string_to_severity(self.skip_trade_threshold))
        scale_threshold_score = self._severity_score(self._string_to_severity(self.scale_down_threshold))

        if severity_score >= skip_threshold_score:
            return AnomalyResponse.SKIP_TRADE
        elif severity_score >= scale_threshold_score:
            return AnomalyResponse.SCALE_DOWN
        else:
            return AnomalyResponse.LOG_ONLY

    def _string_to_severity(self, severity_str: str) -> AnomalySeverity:
        """Convert string to AnomalySeverity enum."""
        mapping = {
            'low': AnomalySeverity.LOW,
            'medium': AnomalySeverity.MEDIUM,
            'high': AnomalySeverity.HIGH,
            'critical': AnomalySeverity.CRITICAL
        }
        return mapping.get(severity_str.lower(), AnomalySeverity.LOW)

    def _log_anomaly(self, symbol: str, anomaly: AnomalyResult, market_data: pd.DataFrame,
                    signal: Optional[Dict[str, Any]] = None, response: Optional[AnomalyResponse] = None):
        """Log anomaly details."""
        try:
            # Create anomaly log entry
            log_entry = AnomalyLog(
                symbol=symbol,
                anomaly_result=anomaly,
                market_data=market_data.tail(5).to_dict('records') if not market_data.empty else None,
                action_taken=response.value if response else None,
                original_signal=signal
            )

            # Add to history
            self.anomaly_history.append(log_entry)
            if len(self.anomaly_history) > self.max_history:
                self.anomaly_history = self.anomaly_history[-self.max_history:]

            # Log to text file
            self._log_to_file(log_entry)

            # Log to JSON file
            self._log_to_json(log_entry)

            # Log to trade logger (use performance method as anomaly method may not exist)
            trade_logger.performance(
                f"Anomaly detected: {anomaly.anomaly_type.value}",
                {
                    'symbol': symbol,
                    'anomaly_type': anomaly.anomaly_type.value,
                    'severity': anomaly.severity.value,
                    'confidence': anomaly.confidence_score,
                    'action': response.value if response else 'none',
                    'z_score': anomaly.z_score,
                    'threshold': anomaly.threshold
                }
            )

        except Exception as e:
            logger.error(f"Failed to log anomaly: {e}")

    def _log_to_file(self, log_entry: AnomalyLog):
        """Log anomaly to text file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

            with open(self.log_file, 'a', encoding='utf-8') as f:
                timestamp = log_entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') if log_entry.timestamp else 'Unknown'
                f.write(f"[{timestamp}] {log_entry.symbol}: {log_entry.anomaly_result.anomaly_type.value} "
                       f"(severity: {log_entry.anomaly_result.severity.value}, "
                       f"confidence: {log_entry.anomaly_result.confidence_score:.3f}) "
                       f"Action: {log_entry.action_taken or 'none'}\n")

        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def _log_to_json(self, log_entry: AnomalyLog):
        """Log anomaly to JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.json_log_file), exist_ok=True)

            # Read existing logs
            existing_logs = []
            if os.path.exists(self.json_log_file):
                try:
                    with open(self.json_log_file, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    existing_logs = []

            # Add new log entry
            existing_logs.append(log_entry.to_dict())

            # Keep only recent logs
            if len(existing_logs) > self.max_history:
                existing_logs = existing_logs[-self.max_history:]

            # Write back to file
            with open(self.json_log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to write to JSON log file: {e}")

    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected anomalies."""
        if not self.anomaly_history:
            return {'total_anomalies': 0}

        # Count by type
        type_counts = {}
        severity_counts = {}
        response_counts = {}

        for log_entry in self.anomaly_history:
            anomaly_type = log_entry.anomaly_result.anomaly_type.value
            severity = log_entry.anomaly_result.severity.value
            response = log_entry.action_taken or 'none'

            type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            response_counts[response] = response_counts.get(response, 0) + 1

        return {
            'total_anomalies': len(self.anomaly_history),
            'by_type': type_counts,
            'by_severity': severity_counts,
            'by_response': response_counts,
            'most_common_type': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            'most_common_severity': max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None
        }

    def clear_history(self):
        """Clear anomaly history."""
        self.anomaly_history = []
        logger.info("Anomaly history cleared")


# Global anomaly detector instance
_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get the global anomaly detector instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        config = get_config('anomaly_detector', {})
        _anomaly_detector = AnomalyDetector(config)
    return _anomaly_detector


def detect_anomalies(data: pd.DataFrame, symbol: str = "") -> List[AnomalyResult]:
    """
    Convenience function to detect anomalies.

    Args:
        data: Market data DataFrame
        symbol: Trading symbol

    Returns:
        List of detected anomalies
    """
    detector = get_anomaly_detector()
    return detector.detect_anomalies(data, symbol)


def check_signal_anomaly(signal: Dict[str, Any], data: pd.DataFrame,
                        symbol: str = "") -> Tuple[bool, Optional[AnomalyResponse], Optional[AnomalyResult]]:
    """
    Convenience function to check signal for anomalies.

    Args:
        signal: Trading signal dictionary
        data: Market data DataFrame
        symbol: Trading symbol

    Returns:
        Tuple of (should_proceed, response_type, anomaly_result)
    """
    detector = get_anomaly_detector()
    return detector.check_signal_anomaly(signal, data, symbol)
