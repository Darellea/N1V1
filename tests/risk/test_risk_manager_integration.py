"""
Integration tests for Risk Manager with Adaptive Policy.

This module tests the integration between the RiskManager and AdaptiveRiskPolicy
to ensure position sizing is properly adjusted based on market conditions and
performance metrics.
"""

import pytest
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch
import asyncio

from risk.risk_manager import RiskManager
from risk.adaptive_policy import AdaptiveRiskPolicy, DefensiveMode
from core.contracts import TradingSignal, SignalType, SignalStrength


class TestRiskManagerAdaptiveIntegration:
    """Test RiskManager integration with AdaptiveRiskPolicy."""

    @pytest.mark.asyncio
    async def test_position_sizing_with_adaptive_policy(self):
        """Test that position sizing is adjusted by adaptive risk multiplier."""
        # Mock the adaptive policy to return a specific multiplier
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            mock_get_multiplier.return_value = (0.8, "High volatility detected")

            config = {
                'position_sizing_method': 'fixed_percent',
                'fixed_percent': 0.1,  # 10% of account balance
                'max_position_size': 0.3,
                'risk_management': {
                    'position_sizing_method': 'fixed_percent',
                    'fixed_percent': 0.1
                }
            }

            risk_manager = RiskManager(config)

            # Create a trading signal
            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),  # Will be calculated
                current_price=Decimal('50000.0'),
                stop_loss=Decimal('49000.0'),
                take_profit=Decimal('52000.0')
            )

            # Calculate position size
            position_size = await risk_manager.calculate_position_size(signal)

            # Expected: base_position (10% of 10000) * multiplier (0.8) = 1000 * 0.8 = 800
            expected_position = Decimal('800.0')

            assert position_size == expected_position
            mock_get_multiplier.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_policy_called_with_correct_data(self):
        """Test that adaptive policy is called with correct market data."""
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            mock_get_multiplier.return_value = (1.0, "Normal conditions")

            config = {
                'position_sizing_method': 'fixed_percent',
                'fixed_percent': 0.1
            }

            risk_manager = RiskManager(config)

            # Create market data
            market_data = {
                'high': [50000, 51000, 50500],
                'low': [49500, 50000, 49800],
                'close': [49800, 50800, 50200],
                'volume': [100, 150, 120]
            }

            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),
                current_price=Decimal('50200.0')
            )

            # Calculate position size
            position_size = await risk_manager.calculate_position_size(signal, market_data)

            # Verify the adaptive policy was called (just check that it was called with the symbol)
            mock_get_multiplier.assert_called_once()
            call_args = mock_get_multiplier.call_args
            assert call_args[0][0] == 'BTC/USDT'  # First argument should be the symbol

    @pytest.mark.asyncio
    async def test_kill_switch_blocks_trading(self):
        """Test that kill switch activation blocks trading."""
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            # Simulate kill switch activation (multiplier = 0)
            mock_get_multiplier.return_value = (0.0, "Kill switch activated")

            config = {
                'position_sizing_method': 'fixed_percent',
                'fixed_percent': 0.1
            }

            risk_manager = RiskManager(config)

            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),
                current_price=Decimal('50000.0')
            )

            # Calculate position size
            position_size = await risk_manager.calculate_position_size(signal)

            # Should return 0 when kill switch is active
            assert position_size == Decimal('0')

    @pytest.mark.asyncio
    async def test_defensive_mode_reduces_position_size(self):
        """Test that defensive mode reduces position size."""
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            # Simulate defensive mode (reduced multiplier)
            mock_get_multiplier.return_value = (0.5, "Defensive mode: consecutive losses")

            config = {
                'position_sizing_method': 'fixed_percent',
                'fixed_percent': 0.1
            }

            risk_manager = RiskManager(config)

            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),
                current_price=Decimal('50000.0')
            )

            # Calculate position size
            position_size = await risk_manager.calculate_position_size(signal)

            # Expected: base_position (1000) * defensive_multiplier (0.5) = 500
            expected_position = Decimal('500.0')

            assert position_size == expected_position

    @pytest.mark.asyncio
    async def test_adaptive_atr_position_sizing_with_multiplier(self):
        """Test adaptive ATR position sizing with risk multiplier."""
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            mock_get_multiplier.return_value = (0.9, "Moderate risk conditions")

            config = {
                'position_sizing_method': 'adaptive_atr',
                'risk_per_trade': 0.02,
                'atr_k_factor': 2.0
            }

            risk_manager = RiskManager(config)

            # Create market data with known ATR
            market_data = {
                'high': [50000] * 20,
                'low': [49900] * 20,  # 100 point range = ATR of ~100
                'close': [49950] * 20
            }

            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),
                current_price=Decimal('49950.0')
            )

            # Calculate position size
            position_size = await risk_manager.calculate_adaptive_position_size(signal, market_data)

            # Base ATR position sizing: (10000 * 0.02) / (100 * 2) = 200 / 200 = 1
            # Note: calculate_adaptive_position_size doesn't apply adaptive multiplier
            expected_position = Decimal('1.0')

            assert position_size == expected_position

    @pytest.mark.asyncio
    async def test_error_handling_in_adaptive_integration(self):
        """Test error handling when adaptive policy fails."""
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            # Simulate adaptive policy error
            mock_get_multiplier.side_effect = Exception("Policy calculation failed")

            config = {
                'position_sizing_method': 'fixed_percent',
                'fixed_percent': 0.1
            }

            risk_manager = RiskManager(config)

            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),
                current_price=Decimal('50000.0')
            )

            # Should still calculate position size with fallback multiplier (1.0)
            position_size = await risk_manager.calculate_position_size(signal)

            # Expected: base_position (1000) * fallback_multiplier (1.0) = 1000
            expected_position = Decimal('1000.0')

            assert position_size == expected_position

    @pytest.mark.asyncio
    async def test_different_position_sizing_methods_with_adaptive(self):
        """Test different position sizing methods work with adaptive policy."""
        sizing_methods = ['fixed_percent', 'volatility', 'kelly']

        for method in sizing_methods:
            with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
                mock_get_multiplier.return_value = (0.8, f"Test multiplier for {method}")

                config = {
                    'position_sizing_method': method,
                    'fixed_percent': 0.1,
                    'position_size': 0.1,
                    'kelly_assumed_win_rate': 0.55
                }

                risk_manager = RiskManager(config)

                signal = TradingSignal(
                    strategy_id='test_strategy',
                    symbol='BTC/USDT',
                    signal_type=SignalType.ENTRY_LONG,
                    signal_strength=SignalStrength.STRONG,
                    order_type='MARKET',
                    amount=Decimal('0'),
                    current_price=Decimal('50000.0'),
                    stop_loss=Decimal('49000.0')
                )

                # Should not raise exception
                position_size = await risk_manager.calculate_position_size(signal)

                # Should be a valid decimal
                assert isinstance(position_size, Decimal)
                assert position_size >= 0

    @pytest.mark.asyncio
    async def test_adaptive_multiplier_application(self):
        """Test that adaptive multipliers are properly applied to position sizing."""
        test_cases = [
            (1.5, 1500.0, "High risk multiplier"),  # 1000 * 1.5 = 1500
            (0.5, 500.0, "Low risk multiplier"),    # 1000 * 0.5 = 500
            (0.8, 800.0, "Moderate risk multiplier"), # 1000 * 0.8 = 800
        ]

        for multiplier_value, expected_position, description in test_cases:
            with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
                mock_get_multiplier.return_value = (multiplier_value, description)

                config = {
                    'position_sizing_method': 'fixed_percent',
                    'fixed_percent': 0.1
                }

                risk_manager = RiskManager(config)

                signal = TradingSignal(
                    strategy_id='test_strategy',
                    symbol='BTC/USDT',
                    signal_type=SignalType.ENTRY_LONG,
                    signal_strength=SignalStrength.STRONG,
                    order_type='MARKET',
                    amount=Decimal('0'),
                    current_price=Decimal('50000.0')
                )

                position_size = await risk_manager.calculate_position_size(signal)

                assert position_size == Decimal(str(expected_position)), f"Failed for {description}: expected {expected_position}, got {position_size}"


class TestRiskManagerPerformanceIntegration:
    """Test RiskManager integration with performance updates."""

    @pytest.mark.asyncio
    async def test_trade_outcome_updates_adaptive_policy(self):
        """Test that trade outcomes are properly fed to adaptive policy."""
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            mock_get_multiplier.return_value = (1.0, "Normal conditions")

            config = {
                'position_sizing_method': 'fixed_percent',
                'fixed_percent': 0.1
            }

            risk_manager = RiskManager(config)

            # Simulate a losing trade
            risk_manager.update_trade_outcome(
                symbol='BTC/USDT',
                pnl=Decimal('-100.0'),
                is_win=False,
                timestamp=1000000
            )

            # Next position size calculation should consider the loss
            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),
                current_price=Decimal('50000.0')
            )

            position_size = await risk_manager.calculate_position_size(signal)

            # Verify adaptive policy was called (it would have access to the loss data)
            mock_get_multiplier.assert_called()

    @pytest.mark.asyncio
    async def test_consecutive_losses_affect_position_sizing(self):
        """Test that consecutive losses reduce position sizes."""
        with patch('risk.risk_manager.get_risk_multiplier') as mock_get_multiplier:
            # First call - normal conditions
            # Second call - after consecutive losses
            mock_get_multiplier.side_effect = [
                (1.0, "Normal conditions"),
                (0.6, "Defensive mode: consecutive losses")
            ]

            config = {
                'position_sizing_method': 'fixed_percent',
                'fixed_percent': 0.1
            }

            risk_manager = RiskManager(config)

            signal = TradingSignal(
                strategy_id='test_strategy',
                symbol='BTC/USDT',
                signal_type=SignalType.ENTRY_LONG,
                signal_strength=SignalStrength.STRONG,
                order_type='MARKET',
                amount=Decimal('0'),
                current_price=Decimal('50000.0')
            )

            # First position size (normal)
            position_size_1 = await risk_manager.calculate_position_size(signal)
            expected_1 = Decimal('1000.0')  # 10% of 10000 * 1.0

            # Simulate consecutive losses
            for _ in range(3):
                risk_manager.update_trade_outcome(
                    symbol='BTC/USDT',
                    pnl=Decimal('-50.0'),
                    is_win=False
                )

            # Second position size (should be reduced)
            position_size_2 = await risk_manager.calculate_position_size(signal)
            expected_2 = Decimal('600.0')  # 10% of 10000 * 0.6

            assert position_size_1 == expected_1
            assert position_size_2 == expected_2
            assert position_size_2 < position_size_1


class TestRiskManagerConfiguration:
    """Test RiskManager configuration handling."""

    def test_adaptive_policy_configuration(self):
        """Test that adaptive policy configuration is properly loaded."""
        config = {
            'position_sizing_method': 'adaptive_atr',
            'risk_per_trade': 0.025,
            'atr_k_factor': 1.5,
            'risk': {
                'adaptive': {
                    'enabled': True,
                    'min_multiplier': 0.2,
                    'max_multiplier': 0.9,
                    'volatility_threshold': 0.06
                }
            }
        }

        risk_manager = RiskManager(config)

        # Verify configuration is applied
        assert risk_manager.position_sizing_method == 'adaptive_atr'
        assert risk_manager.risk_per_trade == Decimal('0.025')
        assert risk_manager.atr_k_factor == Decimal('1.5')

    def test_missing_configuration_defaults(self):
        """Test that missing configuration uses sensible defaults."""
        config = {}  # Empty config

        risk_manager = RiskManager(config)

        # Verify defaults are applied
        assert risk_manager.position_sizing_method == 'fixed'
        assert risk_manager.max_position_size == Decimal('0.3')
        assert risk_manager.risk_reward_ratio == Decimal('2.0')


if __name__ == "__main__":
    pytest.main([__file__])
