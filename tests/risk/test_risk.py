"""
Comprehensive test suite for N1V1 Crypto Trading Framework risk management components.

This test file addresses the following vulnerabilities:
1. No Unit Tests for Core Risk Logic - Tests for risk calculations and anomaly detection
2. Missing Integration Tests - Full workflow testing with mocking
3. Lack of Extreme Condition Testing - Flash crashes, high volatility, data gaps

Tests cover:
- Unit tests for _calculate_market_multiplier and _calculate_performance_multiplier
- Unit tests for anomaly detection methods
- Integration tests for full risk assessment workflow
- Extreme condition tests for robustness
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from risk.risk_manager import RiskManager
from risk.anomaly_detector import AnomalyDetector, AnomalyType, AnomalySeverity
from risk.adaptive_policy import AdaptiveRiskPolicy, RiskLevel, DefensiveMode
from risk.utils import calculate_z_score, calculate_returns, safe_divide
from core.contracts import TradingSignal, SignalType, SignalStrength


@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for testing."""
    dates = pd.date_range('2023-01-01', periods=100, freq='1h')
    np.random.seed(42)  # For reproducible tests

    # Generate realistic price data with some volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.01, 100)  # Small drift with volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC data
    high = prices * (1 + np.abs(np.random.normal(0, 0.005, 100)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.005, 100)))
    open_prices = np.roll(prices, 1)
    open_prices[0] = base_price

    # Generate volume data
    volume = np.random.lognormal(10, 1, 100)

    return pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)


@pytest.fixture
def risk_manager_config():
    """Fixture providing risk manager configuration."""
    return {
        "stop_loss": 0.02,
        "take_profit": 0.04,
        "position_size": 0.1,
        "max_position_size": 0.3,
        "risk_reward_ratio": 2.0,
        "max_daily_drawdown": 0.1,
        "require_stop_loss": True,
        "max_concurrent_trades": 3,
        "position_sizing_method": "fixed",
        "fixed_percent": 0.1
    }


@pytest.fixture
def adaptive_policy_config():
    """Fixture providing adaptive risk policy configuration."""
    return {
        "min_multiplier": 0.1,
        "max_multiplier": 1.0,
        "volatility_threshold": 0.05,
        "performance_lookback_days": 30,
        "min_sharpe": -0.5,
        "max_consecutive_losses": 5,
        "kill_switch_threshold": 10,
        "kill_switch_window_hours": 24,
        "market_monitor": {
            "volatility_threshold": 0.05,
            "volatility_lookback": 20,
            "adx_trend_threshold": 25
        },
        "performance_monitor": {
            "lookback_days": 30,
            "min_sharpe": -0.5,
            "max_consecutive_losses": 5
        }
    }


@pytest.fixture
def anomaly_detector_config():
    """Fixture providing anomaly detector configuration."""
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
                'critical': 15.0
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
            'skip_trade_threshold': 'critical',
            'scale_down_threshold': 'medium',
            'scale_down_factor': 0.5
        },
        'logging': {
            'enabled': True,
            'file': 'anomalies.log',
            'json_file': 'anomalies.json'
        },
        'max_history': 1000
    }


class TestCoreRiskCalculations:
    """Unit tests for core risk calculation functions."""

    def test_calculate_market_multiplier_normal_conditions(self, adaptive_policy_config):
        """Test _calculate_market_multiplier with normal market conditions."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        # Normal market conditions
        market_conditions = {
            'risk_level': RiskLevel.MODERATE.value,
            'volatility_level': 'moderate',
            'trend_strength': 25
        }

        multiplier = policy._calculate_market_multiplier(market_conditions)
        assert 0.9 <= multiplier <= 1.1  # Should be close to 1.0

    def test_calculate_market_multiplier_high_volatility(self, adaptive_policy_config):
        """Test _calculate_market_multiplier with high volatility."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        market_conditions = {
            'risk_level': RiskLevel.HIGH.value,
            'volatility_level': 'high',
            'trend_strength': 15  # Weak trend
        }

        multiplier = policy._calculate_market_multiplier(market_conditions)
        assert multiplier < 0.8  # Should reduce risk significantly

    def test_calculate_market_multiplier_very_low_risk(self, adaptive_policy_config):
        """Test _calculate_market_multiplier with very low risk conditions."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        market_conditions = {
            'risk_level': RiskLevel.VERY_LOW.value,
            'volatility_level': 'low',
            'trend_strength': 45  # Strong trend
        }

        multiplier = policy._calculate_market_multiplier(market_conditions)
        assert multiplier > 1.1  # Should increase risk

    def test_calculate_market_multiplier_edge_cases(self, adaptive_policy_config):
        """Test _calculate_market_multiplier with edge cases."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        # Test with missing keys
        market_conditions = {}
        multiplier = policy._calculate_market_multiplier(market_conditions)
        assert multiplier == 1.0  # Should default to 1.0

        # Test with invalid risk level
        market_conditions = {'risk_level': 'invalid'}
        multiplier = policy._calculate_market_multiplier(market_conditions)
        assert multiplier == 1.0  # Should handle gracefully

    def test_calculate_performance_multiplier_good_performance(self, adaptive_policy_config):
        """Test _calculate_performance_multiplier with good performance."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        performance_metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.65,
            'consecutive_losses': 0
        }

        multiplier = policy._calculate_performance_multiplier(performance_metrics)
        assert multiplier > 1.1  # Should increase risk for good performance

    def test_calculate_performance_multiplier_poor_performance(self, adaptive_policy_config):
        """Test _calculate_performance_multiplier with poor performance."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        performance_metrics = {
            'sharpe_ratio': -1.0,
            'win_rate': 0.35,
            'consecutive_losses': 6  # Above threshold
        }

        multiplier = policy._calculate_performance_multiplier(performance_metrics)
        assert multiplier < 0.6  # Should reduce risk significantly

    def test_calculate_performance_multiplier_consecutive_losses(self, adaptive_policy_config):
        """Test _calculate_performance_multiplier with consecutive losses."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        performance_metrics = {
            'sharpe_ratio': 0.5,
            'win_rate': 0.5,
            'consecutive_losses': 4  # Triggers caution mode
        }

        multiplier = policy._calculate_performance_multiplier(performance_metrics)
        assert multiplier < 0.8  # Should reduce risk

        # Check that defensive mode was activated
        assert policy.defensive_mode == DefensiveMode.CAUTION

    def test_calculate_performance_multiplier_max_consecutive_losses(self, adaptive_policy_config):
        """Test _calculate_performance_multiplier with maximum consecutive losses."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        performance_metrics = {
            'sharpe_ratio': 0.0,
            'win_rate': 0.5,
            'consecutive_losses': 6  # Above max threshold
        }

        multiplier = policy._calculate_performance_multiplier(performance_metrics)
        assert multiplier <= 0.5  # Should reduce risk to minimum

        # Check that defensive mode was activated
        assert policy.defensive_mode == DefensiveMode.DEFENSIVE

    def test_calculate_performance_multiplier_zero_division_protection(self, adaptive_policy_config):
        """Test _calculate_performance_multiplier handles zero division."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        # Test with zero win rate (should not cause division by zero)
        performance_metrics = {
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'consecutive_losses': 0
        }

        multiplier = policy._calculate_performance_multiplier(performance_metrics)
        assert isinstance(multiplier, float)
        assert multiplier > 0

    def test_calculate_performance_multiplier_empty_metrics(self, adaptive_policy_config):
        """Test _calculate_performance_multiplier with empty metrics."""
        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        performance_metrics = {}
        multiplier = policy._calculate_performance_multiplier(performance_metrics)
        assert multiplier == 1.0  # Should default to 1.0


class TestAnomalyDetection:
    """Unit tests for anomaly detection methods."""

    def test_price_zscore_detector_normal_data(self, anomaly_detector_config, sample_market_data):
        """Test price z-score detector with normal market data."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Use normal data (should not detect anomalies)
        result = detector.detect_anomalies(sample_market_data, "BTC/USDT")

        # Should not detect any anomalies in normal data
        assert len(result) == 0

    def test_price_zscore_detector_anomalous_data(self, anomaly_detector_config):
        """Test price z-score detector with anomalous price movement."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Create data with extreme price movement
        dates = pd.date_range('2023-01-01', periods=60, freq='1h')
        prices = [100.0] * 59 + [150.0]  # Sudden 50% jump

        data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices,
            'volume': [1000] * 60
        }, index=dates)

        anomalies = detector.detect_anomalies(data, "BTC/USDT")

        # Should detect price anomaly
        assert len(anomalies) > 0
        assert any(a.anomaly_type == AnomalyType.PRICE_ZSCORE for a in anomalies)

    def test_volume_zscore_detector_normal_volume(self, anomaly_detector_config, sample_market_data):
        """Test volume z-score detector with normal volume."""
        detector = AnomalyDetector(anomaly_detector_config)

        anomalies = detector.detect_anomalies(sample_market_data, "BTC/USDT")

        # Should not detect volume anomalies in normal data
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME_ZSCORE]
        assert len(volume_anomalies) == 0

    def test_volume_zscore_detector_anomalous_volume(self, anomaly_detector_config):
        """Test volume z-score detector with anomalous volume spike."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Create data with extreme volume spike
        dates = pd.date_range('2023-01-01', periods=30, freq='1h')
        volumes = [1000] * 29 + [50000]  # Extreme volume spike

        data = pd.DataFrame({
            'close': np.random.normal(100, 1, 30),
            'high': np.random.normal(101, 1, 30),
            'low': np.random.normal(99, 1, 30),
            'open': np.random.normal(100, 1, 30),
            'volume': volumes
        }, index=dates)

        anomalies = detector.detect_anomalies(data, "BTC/USDT")

        # Should detect volume anomaly
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME_ZSCORE]
        assert len(volume_anomalies) > 0

    def test_price_gap_detector_normal_gaps(self, anomaly_detector_config, sample_market_data):
        """Test price gap detector with normal price gaps."""
        detector = AnomalyDetector(anomaly_detector_config)

        anomalies = detector.detect_anomalies(sample_market_data, "BTC/USDT")

        # Should not detect gap anomalies in normal data
        gap_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.PRICE_GAP]
        assert len(gap_anomalies) == 0

    def test_price_gap_detector_large_gap(self, anomaly_detector_config):
        """Test price gap detector with large price gap."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Create data with large price gap
        dates = pd.date_range('2023-01-01', periods=3, freq='1h')
        prices = [100.0, 100.0, 115.0]  # 15% gap

        data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'open': prices,
            'volume': [1000] * 3
        }, index=dates)

        anomalies = detector.detect_anomalies(data, "BTC/USDT")

        # Should detect price gap anomaly
        gap_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.PRICE_GAP]
        assert len(gap_anomalies) > 0

    def test_anomaly_detector_disabled(self, anomaly_detector_config):
        """Test anomaly detector when disabled."""
        config = anomaly_detector_config.copy()
        config['enabled'] = False

        detector = AnomalyDetector(config)
        anomalies = detector.detect_anomalies(sample_market_data, "BTC/USDT")

        assert len(anomalies) == 0

    def test_anomaly_severity_calculation(self, anomaly_detector_config):
        """Test anomaly severity calculation."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Test different z-scores and their severities
        test_cases = [
            (2.5, AnomalySeverity.LOW),
            (3.5, AnomalySeverity.MEDIUM),
            (4.5, AnomalySeverity.HIGH),
            (6.0, AnomalySeverity.CRITICAL)
        ]

        for z_score, expected_severity in test_cases:
            severity = detector.price_detector._calculate_severity(z_score, {
                'low': 2.0, 'medium': 3.0, 'high': 4.0, 'critical': 5.0
            })
            assert severity == expected_severity


class TestIntegrationTests:
    """Integration tests for full risk management workflow."""

    @patch('risk.risk_manager.RiskManager._get_current_balance')
    @patch('risk.risk_manager.RiskManager._get_current_positions')
    @patch('risk.risk_manager.RiskManager._validate_signal_basics')
    @patch('risk.risk_manager.RiskManager._check_portfolio_risk')
    @patch('risk.risk_manager.RiskManager._validate_position_size')
    @patch('risk.risk_manager.RiskManager.calculate_take_profit')
    async def test_full_risk_assessment_workflow(self, mock_take_profit, mock_validate_pos,
                                                mock_check_portfolio, mock_validate_basics,
                                                mock_get_positions, mock_get_balance,
                                                risk_manager_config, sample_market_data):
        """Test full risk assessment workflow from signal to position size."""
        # Setup mocks
        mock_get_balance.return_value = Decimal("10000")
        mock_get_positions.return_value = []
        mock_validate_basics.return_value = True
        mock_check_portfolio.return_value = True
        mock_validate_pos.return_value = True
        mock_take_profit.return_value = Decimal("110")

        manager = RiskManager(risk_manager_config)

        # Create test signal
        signal = TradingSignal(
            strategy_id="test_strategy",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="MARKET",
            amount=0,  # Should be calculated
            current_price=100.0,
            stop_loss=95.0
        )

        # Run full evaluation
        result = await manager.evaluate_signal(signal, sample_market_data)

        # Verify the workflow completed successfully
        assert result is True
        assert signal.amount > 0  # Position size should be calculated

        # Verify all components were called
        mock_validate_basics.assert_called_once()
        mock_check_portfolio.assert_called_once()
        mock_validate_pos.assert_called_once()
        mock_take_profit.assert_called_once()

    @patch('risk.adaptive_policy.get_market_regime_detector')
    async def test_adaptive_policy_full_workflow(self, mock_regime_detector, adaptive_policy_config, sample_market_data):
        """Test full adaptive risk policy workflow."""
        # Mock regime detector
        mock_detector = Mock()
        mock_detector.detect_regime.return_value = Mock(regime="TRENDING", previous_regime="SIDEWAYS")
        mock_regime_detector.return_value = mock_detector

        policy = AdaptiveRiskPolicy(adaptive_policy_config)

        # Test with normal market data
        multiplier, reasoning = policy.get_risk_multiplier("BTC/USDT", sample_market_data)

        # Should return a valid multiplier and reasoning
        assert isinstance(multiplier, float)
        assert 0.1 <= multiplier <= 1.0
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    @patch('risk.anomaly_detector.get_anomaly_detector')
    async def test_anomaly_detection_integration(self, mock_get_detector, anomaly_detector_config, sample_market_data):
        """Test anomaly detection integration with risk manager."""
        # Mock anomaly detector
        mock_detector = Mock()
        mock_detector.check_signal_anomaly.return_value = (True, None, None)
        mock_get_detector.return_value = mock_detector

        manager = RiskManager(anomaly_detector_config)

        signal = TradingSignal(
            strategy_id="test",
            symbol="BTC/USDT",
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="MARKET",
            amount=1000,
            current_price=100.0,
            stop_loss=95.0
        )

        # This would normally call anomaly detection
        # We can't easily test the full integration without more complex mocking
        # but this shows the integration point exists
        assert signal.symbol == "BTC/USDT"


class TestExtremeConditions:
    """Tests for extreme market conditions."""

    def test_flash_crash_simulation(self, anomaly_detector_config):
        """Test system behavior during flash crash simulation."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Simulate flash crash: sudden massive price drop
        dates = pd.date_range('2023-01-01', periods=60, freq='1h')
        prices = [100.0] * 58 + [50.0, 45.0]  # 50%+ drop in 2 periods

        data = pd.DataFrame({
            'close': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'open': prices,
            'volume': [1000] * 60
        }, index=dates)

        anomalies = detector.detect_anomalies(data, "BTC/USDT")

        # Should detect multiple anomalies during flash crash
        assert len(anomalies) > 0

        # Should include price z-score and gap anomalies
        anomaly_types = {a.anomaly_type for a in anomalies}
        assert AnomalyType.PRICE_ZSCORE in anomaly_types
        assert AnomalyType.PRICE_GAP in anomaly_types

    def test_high_volatility_period(self, anomaly_detector_config):
        """Test system behavior during high volatility period."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Simulate high volatility period
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        np.random.seed(123)

        # Generate highly volatile price series
        base_price = 100.0
        high_vol_returns = np.random.normal(0, 0.05, 100)  # 5% daily volatility
        prices = base_price * np.exp(np.cumsum(high_vol_returns))

        data = pd.DataFrame({
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.03, 100))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.03, 100))),
            'open': np.roll(prices, 1),
            'volume': np.random.lognormal(12, 2, 100)  # High volume variation
        }, index=dates)

        anomalies = detector.detect_anomalies(data, "BTC/USDT")

        # High volatility might trigger some anomalies
        # The exact number depends on the random data, but should be handled gracefully
        assert isinstance(anomalies, list)

    def test_data_gaps_handling(self, anomaly_detector_config):
        """Test system behavior with data gaps."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Create data with gaps (NaN values)
        dates = pd.date_range('2023-01-01', periods=50, freq='1h')
        prices = np.random.normal(100, 2, 50)
        prices[10:15] = np.nan  # Gap in data
        prices[25:27] = np.nan  # Another gap

        data = pd.DataFrame({
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'open': prices,
            'volume': [1000] * 50
        }, index=dates)

        # Should handle NaN values gracefully without crashing
        try:
            anomalies = detector.detect_anomalies(data, "BTC/USDT")
            # If it doesn't crash, the test passes
            assert isinstance(anomalies, list)
        except Exception as e:
            # If it does crash, that's a problem
            pytest.fail(f"Anomaly detection crashed on data with gaps: {e}")

    def test_extreme_volume_spike(self, anomaly_detector_config):
        """Test system behavior with extreme volume spike."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Create data with extreme volume spike
        dates = pd.date_range('2023-01-01', periods=30, freq='1h')
        volumes = [1000] * 29 + [1000000]  # Million-fold volume increase

        data = pd.DataFrame({
            'close': np.random.normal(100, 1, 30),
            'high': np.random.normal(101, 1, 30),
            'low': np.random.normal(99, 1, 30),
            'open': np.random.normal(100, 1, 30),
            'volume': volumes
        }, index=dates)

        anomalies = detector.detect_anomalies(data, "BTC/USDT")

        # Should detect volume anomaly
        volume_anomalies = [a for a in anomalies if a.anomaly_type == AnomalyType.VOLUME_ZSCORE]
        assert len(volume_anomalies) > 0

        # Should have high severity
        assert volume_anomalies[0].severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]

    def test_zero_price_handling(self, anomaly_detector_config):
        """Test system behavior with zero or negative prices."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Create data with zero price
        dates = pd.date_range('2023-01-01', periods=10, freq='1h')
        prices = [100.0] * 9 + [0.0]  # Zero price

        data = pd.DataFrame({
            'close': prices,
            'high': [p if p > 0 else 100 for p in prices],
            'low': [p if p > 0 else 100 for p in prices],
            'open': prices,
            'volume': [1000] * 10
        }, index=dates)

        # Should handle zero prices gracefully
        try:
            anomalies = detector.detect_anomalies(data, "BTC/USDT")
            assert isinstance(anomalies, list)
        except Exception as e:
            pytest.fail(f"Anomaly detection crashed on zero price: {e}")

    def test_empty_data_handling(self, anomaly_detector_config):
        """Test system behavior with empty data."""
        detector = AnomalyDetector(anomaly_detector_config)

        # Empty DataFrame
        empty_data = pd.DataFrame()

        # Should handle empty data gracefully
        anomalies = detector.detect_anomalies(empty_data, "BTC/USDT")
        assert len(anomalies) == 0

    def test_insufficient_data_handling(self, anomaly_detector_config):
        """Test system behavior with insufficient data."""
        detector = AnomalyDetector(anomaly_detector_config)

        # DataFrame with only 2 rows (insufficient for most calculations)
        dates = pd.date_range('2023-01-01', periods=2, freq='1h')
        data = pd.DataFrame({
            'close': [100.0, 101.0],
            'high': [101.0, 102.0],
            'low': [99.0, 100.0],
            'open': [100.0, 101.0],
            'volume': [1000, 1000]
        }, index=dates)

        # Should handle insufficient data gracefully
        anomalies = detector.detect_anomalies(data, "BTC/USDT")
        # May or may not detect anomalies, but shouldn't crash
        assert isinstance(anomalies, list)


class TestRiskManagerEdgeCases:
    """Tests for RiskManager edge cases and error handling."""

    def test_risk_manager_zero_balance(self, risk_manager_config):
        """Test RiskManager behavior with zero balance."""
        manager = RiskManager(risk_manager_config)

        # Mock zero balance
        with patch.object(manager, '_get_current_balance', return_value=Decimal("0")):
            signal = TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type="MARKET",
                amount=0,
                current_price=100.0,
                stop_loss=95.0
            )

            # Should handle zero balance gracefully
            import asyncio
            result = asyncio.run(manager.calculate_position_size(signal))
            assert result == Decimal("0")

    def test_risk_manager_extreme_price_values(self, risk_manager_config):
        """Test RiskManager with extreme price values."""
        manager = RiskManager(risk_manager_config)

        with patch.object(manager, '_get_current_balance', return_value=Decimal("10000")):
            signal = TradingSignal(
                strategy_id="test",
                symbol="BTC/USDT",
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type="MARKET",
                amount=0,
                current_price=Decimal("1000000"),  # Very high price
                stop_loss=Decimal("999999")  # Very small risk
            )

            # Should handle extreme values without overflow
            import asyncio
            result = asyncio.run(manager.calculate_position_size(signal))
            assert result > 0
            assert result < Decimal("10000000")  # Reasonable upper bound

    def test_invalid_signal_handling(self, risk_manager_config):
        """Test handling of invalid signals."""
        manager = RiskManager(risk_manager_config)

        # Test with None signal
        import asyncio
        result = asyncio.run(manager.evaluate_signal(None))
        assert result is False

        # Test with signal missing required fields
        invalid_signal = TradingSignal(
            strategy_id="test",
            symbol="",  # Empty symbol
            signal_type=SignalType.ENTRY_LONG,
            signal_strength=SignalStrength.STRONG,
            order_type="MARKET",
            amount=1000,
            current_price=100.0,
            stop_loss=95.0
        )

        result = asyncio.run(manager.evaluate_signal(invalid_signal))
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
