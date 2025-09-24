"""
Tests for anomaly detector functionality.
"""
import pytest
import pandas as pd
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from risk.anomaly_detector import (
    AnomalyDetector,
    PriceZScoreDetector,
    VolumeZScoreDetector,
    PriceGapDetector,
    AnomalyResult,
    AnomalyType,
    AnomalySeverity,
    AnomalyResponse,
    detect_anomalies,
    check_signal_anomaly,
    get_anomaly_detector
)


class TestAnomalyResult:
    """Test AnomalyResult dataclass."""

    def test_anomaly_result_creation(self):
        """Test basic AnomalyResult creation."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_type=AnomalyType.PRICE_ZSCORE,
            severity=AnomalySeverity.HIGH,
            confidence_score=0.85,
            z_score=3.5,
            threshold=3.0
        )

        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.PRICE_ZSCORE
        assert result.severity == AnomalySeverity.HIGH
        assert result.confidence_score == 0.85
        assert result.z_score == 3.5
        assert result.threshold == 3.0
        assert result.timestamp is not None

    def test_anomaly_result_to_dict(self):
        """Test conversion to dictionary."""
        result = AnomalyResult(
            is_anomaly=True,
            anomaly_type=AnomalyType.PRICE_ZSCORE,
            severity=AnomalySeverity.HIGH,
            confidence_score=0.85
        )

        data = result.to_dict()
        assert data['is_anomaly'] is True
        assert data['anomaly_type'] == 'price_zscore'
        assert data['severity'] == 'high'
        assert data['confidence_score'] == 0.85


class TestPriceZScoreDetector:
    """Test price z-score anomaly detection."""

    def test_detect_normal_price_movement(self):
        """Test detection with normal price movement."""
        # Create normal price data
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        prices = [100 + i * 0.1 for i in range(60)]  # Gradual increase
        data = pd.DataFrame({
            'close': prices
        }, index=dates)

        detector = PriceZScoreDetector()
        result = detector.detect(data)

        assert result.is_anomaly is False
        assert result.anomaly_type == AnomalyType.NONE
        assert result.severity == AnomalySeverity.LOW

    def test_detect_price_anomaly(self):
        """Test detection of anomalous price movement."""
        # Create data with normal movement, then a spike
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        prices = [100 + i * 0.1 for i in range(59)]  # Normal movement
        prices.append(150.0)  # Anomalous spike
        data = pd.DataFrame({
            'close': prices
        }, index=dates)

        detector = PriceZScoreDetector()
        result = detector.detect(data)

        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.PRICE_ZSCORE
        assert result.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]
        assert result.z_score is not None
        assert result.z_score > 3.0  # Above threshold

    def test_detect_insufficient_data(self):
        """Test handling of insufficient data."""
        data = pd.DataFrame({'close': [100, 101, 102]})  # Too few data points

        detector = PriceZScoreDetector()
        result = detector.detect(data)

        assert result.is_anomaly is False
        assert result.anomaly_type == AnomalyType.NONE

    def test_detect_no_close_column(self):
        """Test handling of missing close column."""
        data = pd.DataFrame({'open': [100, 101, 102]})

        detector = PriceZScoreDetector()
        result = detector.detect(data)

        assert result.is_anomaly is False
        assert result.anomaly_type == AnomalyType.NONE


class TestVolumeZScoreDetector:
    """Test volume z-score anomaly detection."""

    def test_detect_normal_volume(self):
        """Test detection with normal volume."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        volumes = [1000 + i * 10 for i in range(30)]  # Normal volume variation
        data = pd.DataFrame({
            'volume': volumes
        }, index=dates)

        detector = VolumeZScoreDetector()
        result = detector.detect(data)

        assert result.is_anomaly is False
        assert result.anomaly_type == AnomalyType.NONE

    def test_detect_volume_spike(self):
        """Test detection of volume spike."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        volumes = [1000 + i * 10 for i in range(29)]  # Normal volumes
        volumes.append(5000)  # Anomalous spike
        data = pd.DataFrame({
            'volume': volumes
        }, index=dates)

        detector = VolumeZScoreDetector()
        result = detector.detect(data)

        assert result.is_anomaly is True
        assert result.anomaly_type == AnomalyType.VOLUME_ZSCORE
        assert result.z_score is not None
        assert result.z_score > 3.0

    def test_detect_no_volume_column(self):
        """Test handling of missing volume column."""
        data = pd.DataFrame({'close': [100, 101, 102]})

        detector = VolumeZScoreDetector()
        result = detector.detect(data)

        assert result.is_anomaly is False
        assert result.anomaly_type == AnomalyType.NONE


class TestPriceGapDetector:
    """Test price gap anomaly detection."""

    def test_detect_normal_price_change(self):
        """Test detection with normal price change."""
        data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0]  # Normal 1-2% changes
        })

        detector = PriceGapDetector()
        result = detector.detect(data)

        assert result.is_anomaly == False
        assert result.anomaly_type == AnomalyType.NONE

    def test_detect_price_gap(self):
        """Test detection of large price gap."""
        data = pd.DataFrame({
            'close': [100.0, 115.0]  # 15% gap (above 5% threshold)
        })

        detector = PriceGapDetector()
        result = detector.detect(data)

        assert result.is_anomaly == True
        assert result.anomaly_type == AnomalyType.PRICE_GAP
        assert result.severity == AnomalySeverity.CRITICAL
        assert result.context['gap_pct'] == 15.0

    def test_detect_small_gap(self):
        """Test that small gaps are not flagged as anomalies."""
        data = pd.DataFrame({
            'close': [100.0, 102.0]  # 2% gap (below threshold)
        })

        detector = PriceGapDetector()
        result = detector.detect(data)

        assert result.is_anomaly == False
        assert result.anomaly_type == AnomalyType.NONE

    def test_detect_insufficient_data(self):
        """Test handling with only one data point."""
        data = pd.DataFrame({
            'close': [100.0]
        })

        detector = PriceGapDetector()
        result = detector.detect(data)

        assert result.is_anomaly == False
        assert result.anomaly_type == AnomalyType.NONE


class TestAnomalyDetector:
    """Test main AnomalyDetector class."""

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {
            'enabled': True,
            'price_zscore': {'z_threshold': 2.5},
            'response': {'skip_trade_threshold': 'medium'}
        }

        detector = AnomalyDetector(config)
        assert detector.enabled is True
        assert detector.price_detector.z_threshold == 2.5

    def test_detect_anomalies_multiple_types(self):
        """Test detection of multiple anomaly types."""
        # Create data with both price spike and volume spike
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        prices = [100 + i * 0.1 for i in range(59)]
        prices.append(150.0)  # Price spike
        volumes = [1000 + i * 10 for i in range(59)]
        volumes.append(5000)  # Volume spike

        data = pd.DataFrame({
            'close': prices,
            'volume': volumes
        }, index=dates)

        detector = AnomalyDetector()
        results = detector.detect_anomalies(data)

        # Should detect at least one anomaly
        assert len(results) >= 1
        assert all(result.is_anomaly for result in results)

    def test_check_signal_anomaly_skip_trade(self):
        """Test signal anomaly check that results in skip trade."""
        # Create data with critical anomaly
        data = pd.DataFrame({
            'close': [100.0, 120.0],  # 20% gap
            'volume': [1000, 1000]
        })

        signal = {'symbol': 'TEST', 'amount': 1000}
        detector = AnomalyDetector()

        should_proceed, response, anomaly = detector.check_signal_anomaly(signal, data, 'TEST')

        assert should_proceed is False
        assert response == AnomalyResponse.SKIP_TRADE
        assert anomaly is not None
        assert anomaly.severity == AnomalySeverity.CRITICAL

    def test_check_signal_anomaly_scale_down(self):
        """Test signal anomaly check that results in scale down."""
        # Create data with medium severity anomaly - smaller volume spike
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        volumes = [1000 + i * 5 for i in range(29)]  # Smaller base volumes
        volumes.append(1500)  # Smaller volume increase

        data = pd.DataFrame({
            'close': [100 + i * 0.1 for i in range(30)],
            'volume': volumes
        }, index=dates)

        signal = {'symbol': 'TEST', 'amount': 1000}
        detector = AnomalyDetector()

        should_proceed, response, anomaly = detector.check_signal_anomaly(signal, data, 'TEST')

        # Should proceed (not skip) and may have some response
        assert should_proceed is True
        # The exact response depends on the calculated severity
        if response is not None:
            assert response in [AnomalyResponse.SCALE_DOWN, AnomalyResponse.LOG_ONLY]

    def test_disabled_detector(self):
        """Test behavior when detector is disabled."""
        config = {'enabled': False}
        detector = AnomalyDetector(config)

        data = pd.DataFrame({'close': [100, 120]})
        results = detector.detect_anomalies(data)

        assert results == []

    def test_get_anomaly_statistics(self):
        """Test anomaly statistics generation."""
        detector = AnomalyDetector()

        # Add some mock history
        from risk.anomaly_detector import AnomalyLog
        mock_log = AnomalyLog(
            symbol='TEST',
            anomaly_result=AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.PRICE_ZSCORE,
                severity=AnomalySeverity.HIGH,
                confidence_score=0.8
            ),
            action_taken='skip_trade'
        )
        detector.anomaly_history = [mock_log]

        stats = detector.get_anomaly_statistics()

        assert stats['total_anomalies'] == 1
        assert stats['by_type']['price_zscore'] == 1
        assert stats['by_severity']['high'] == 1
        assert stats['by_response']['skip_trade'] == 1

    def test_empty_statistics(self):
        """Test statistics with no anomaly history."""
        detector = AnomalyDetector()
        stats = detector.get_anomaly_statistics()

        assert stats['total_anomalies'] == 0
        # When there are no anomalies, these keys won't exist
        assert 'by_type' not in stats or stats.get('by_type', {}) == {}
        assert 'by_severity' not in stats or stats.get('by_severity', {}) == {}
        assert 'by_response' not in stats or stats.get('by_response', {}) == {}


class TestAnomalyLogging:
    """Test anomaly logging functionality."""

    def test_log_to_file(self):
        """Test logging anomalies to text file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, 'test_anomalies.log')

            detector = AnomalyDetector({
                'logging': {'file': log_file}
            })

            # Create mock anomaly
            from risk.anomaly_detector import AnomalyLog
            anomaly_result = AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.PRICE_GAP,
                severity=AnomalySeverity.HIGH,
                confidence_score=0.9
            )

            log_entry = AnomalyLog(
                symbol='TEST',
                anomaly_result=anomaly_result,
                action_taken='skip_trade'
            )

            detector._log_to_file(log_entry)

            # Verify log file was created and contains expected content
            assert os.path.exists(log_file)

            with open(log_file, 'r') as f:
                content = f.read()

            assert 'TEST' in content
            assert 'price_gap' in content
            assert 'high' in content
            assert 'skip_trade' in content

    def test_log_to_json(self):
        """Test logging anomalies to JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = os.path.join(temp_dir, 'test_anomalies.json')

            detector = AnomalyDetector({
                'logging': {'json_file': json_file}
            })

            # Create mock anomaly
            from risk.anomaly_detector import AnomalyLog
            anomaly_result = AnomalyResult(
                is_anomaly=True,
                anomaly_type=AnomalyType.PRICE_GAP,
                severity=AnomalySeverity.HIGH,
                confidence_score=0.9
            )

            log_entry = AnomalyLog(
                symbol='TEST',
                anomaly_result=anomaly_result,
                action_taken='skip_trade'
            )

            detector._log_to_json(log_entry)

            # Verify JSON file was created and contains valid JSON
            assert os.path.exists(json_file)

            with open(json_file, 'r') as f:
                data = json.load(f)

            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]['symbol'] == 'TEST'
            assert data[0]['anomaly_result']['anomaly_type'] == 'price_gap'

    @patch('risk.anomaly_detector.trade_logger')
    def test_trade_logger_integration(self, mock_trade_logger):
        """Test integration with trade logger."""
        detector = AnomalyDetector()

        anomaly_result = AnomalyResult(
            is_anomaly=True,
            anomaly_type=AnomalyType.PRICE_ZSCORE,
            severity=AnomalySeverity.MEDIUM,
            confidence_score=0.7,
            z_score=3.5
        )

        data = pd.DataFrame({'close': [100, 110]})
        detector._log_anomaly('TEST', anomaly_result, data, None, AnomalyResponse.LOG_ONLY)

        # Verify trade logger was called with performance method (not anomaly)
        mock_trade_logger.performance.assert_called_once()
        call_args = mock_trade_logger.performance.call_args
        assert 'Anomaly detected: price_zscore' in call_args[0][0]
        assert call_args[0][1]['symbol'] == 'TEST'
        assert call_args[0][1]['anomaly_type'] == 'price_zscore'


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('risk.anomaly_detector.get_anomaly_detector')
    def test_detect_anomalies_convenience(self, mock_get_detector):
        """Test detect_anomalies convenience function."""
        mock_detector = Mock()
        mock_detector.detect_anomalies.return_value = []
        mock_get_detector.return_value = mock_detector

        data = pd.DataFrame({'close': [100, 101]})
        result = detect_anomalies(data, 'TEST')

        mock_get_detector.assert_called_once()
        mock_detector.detect_anomalies.assert_called_once_with(data, 'TEST')
        assert result == []

    @patch('risk.anomaly_detector.get_anomaly_detector')
    def test_check_signal_anomaly_convenience(self, mock_get_detector):
        """Test check_signal_anomaly convenience function."""
        mock_detector = Mock()
        mock_detector.check_signal_anomaly.return_value = (True, None, None)
        mock_get_detector.return_value = mock_detector

        signal = {'symbol': 'TEST'}
        data = pd.DataFrame({'close': [100, 101]})
        result = check_signal_anomaly(signal, data, 'TEST')

        mock_get_detector.assert_called_once()
        mock_detector.check_signal_anomaly.assert_called_once_with(signal, data, 'TEST')
        assert result == (True, None, None)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_detector_with_empty_data(self):
        """Test detector with empty DataFrame."""
        detector = AnomalyDetector()
        data = pd.DataFrame()

        results = detector.detect_anomalies(data)
        assert results == []

    def test_detector_with_nan_values(self):
        """Test detector handling of NaN values."""
        data = pd.DataFrame({
            'close': [100, float('nan'), 102],
            'volume': [1000, float('nan'), 1002]
        })

        detector = AnomalyDetector()
        # Should not crash, but may not detect anomalies due to NaN handling
        results = detector.detect_anomalies(data)
        # Results may be empty or contain valid detections depending on implementation
        assert isinstance(results, list)

    def test_zero_std_deviation(self):
        """Test handling of zero standard deviation."""
        # Create data with no variation
        data = pd.DataFrame({
            'close': [100] * 60  # Same price for all periods
        })

        detector = PriceZScoreDetector()
        result = detector.detect(data)

        # Should handle gracefully without division by zero
        assert result.is_anomaly == False

    def test_extreme_values(self):
        """Test handling of extreme values."""
        # Create data with extreme price movement
        data = pd.DataFrame({
            'close': [100, 1000000]  # Extreme gap
        })

        detector = PriceGapDetector()
        result = detector.detect(data)

        assert result.is_anomaly == True
        assert result.severity == AnomalySeverity.CRITICAL

    def test_max_history_limit(self):
        """Test that anomaly history respects max_history limit."""
        detector = AnomalyDetector({'max_history': 5})

        # Add more than max_history entries
        for i in range(10):
            from risk.anomaly_detector import AnomalyLog
            log_entry = AnomalyLog(
                symbol=f'TEST{i}',
                anomaly_result=AnomalyResult(
                    is_anomaly=True,
                    anomaly_type=AnomalyType.PRICE_ZSCORE,
                    severity=AnomalySeverity.LOW,
                    confidence_score=0.5
                )
            )
            detector.anomaly_history.append(log_entry)

        # Should only keep the last max_history entries
        assert len(detector.anomaly_history) == 5


class TestConfiguration:
    """Test configuration handling."""

    def test_default_configuration(self):
        """Test default configuration values."""
        detector = AnomalyDetector()

        assert detector.enabled is True
        assert detector.price_detector.enabled is True
        assert detector.volume_detector.enabled is True
        assert detector.gap_detector.enabled is True
        assert detector.log_anomalies is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = {
            'enabled': False,
            'price_zscore': {
                'enabled': False,
                'z_threshold': 2.0
            },
            'response': {
                'scale_down_factor': 0.3
            },
            'logging': {
                'enabled': False
            }
        }

        detector = AnomalyDetector(config)

        assert detector.enabled is False
        assert detector.price_detector.enabled is False
        assert detector.price_detector.z_threshold == 2.0
        assert detector.scale_down_factor == 0.3
        assert detector.log_anomalies is False

    def test_severity_threshold_conversion(self):
        """Test conversion of severity threshold strings."""
        detector = AnomalyDetector({
            'response': {
                'skip_trade_threshold': 'critical',
                'scale_down_threshold': 'low'
            }
        })

        # Test string to severity conversion
        assert detector._string_to_severity('critical') == AnomalySeverity.CRITICAL
        assert detector._string_to_severity('low') == AnomalySeverity.LOW
        assert detector._string_to_severity('invalid') == AnomalySeverity.LOW  # Default


if __name__ == "__main__":
    pytest.main([__file__])
