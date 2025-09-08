"""
Comprehensive Circuit Breaker Testing Suite
===========================================

This module provides comprehensive testing for the Circuit Breaker feature,
covering core functionality, state management, integration, performance,
and edge cases as specified in the testing strategy.
"""

import asyncio
import time
import pytest
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import tempfile
import json
import os
from pathlib import Path

from core.circuit_breaker import (
    CircuitBreaker, CircuitBreakerState, CircuitBreakerConfig,
    CircuitBreakerTrigger, CircuitBreakerEvent, get_circuit_breaker
)
from core.order_manager import OrderManager
from core.signal_router import SignalRouter
from core.risk.risk_manager import RiskManager
from core.anomaly_detector import AnomalyDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class TestCircuitBreakerCoreFunctionality:
    """Test core circuit breaker functionality and trigger conditions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig(
            equity_drawdown_threshold=0.1,  # 10%
            consecutive_losses_threshold=5,
            volatility_spike_threshold=0.05,  # 5%
            max_triggers_per_hour=3,
            monitoring_window_minutes=60,
            cooling_period_minutes=5,
            recovery_period_minutes=10
        )
        self.cb = CircuitBreaker(self.config)

    def test_initialization(self):
        """Test circuit breaker initialization."""
        assert self.cb.state == CircuitBreakerState.NORMAL
        assert self.cb.config == self.config
        assert len(self.cb.trigger_history) == 0
        assert len(self.cb.event_history) == 0

    def test_equity_drawdown_trigger(self):
        """Test equity drawdown threshold triggering."""
        # Test normal equity levels
        assert not self.cb._check_equity_drawdown(10000, 9500)  # 5% drawdown

        # Test trigger threshold
        assert self.cb._check_equity_drawdown(10000, 8900)  # 11% drawdown

        # Test extreme drawdown
        assert self.cb._check_equity_drawdown(10000, 5000)  # 50% drawdown

    def test_consecutive_losses_trigger(self):
        """Test consecutive losses threshold triggering."""
        # Add some wins
        for i in range(3):
            self.cb._record_trade_result(100, True)  # Win

        # Should not trigger yet
        assert not self.cb._check_consecutive_losses()

        # Add losses to reach threshold
        for i in range(5):
            self.cb._record_trade_result(100, False)  # Loss

        # Should trigger
        assert self.cb._check_consecutive_losses()

    def test_volatility_spike_trigger(self):
        """Test volatility spike detection."""
        # Normal volatility
        prices = np.array([100, 101, 99, 102, 98])
        assert not self.cb._check_volatility_spike(prices)

        # High volatility spike
        prices = np.array([100, 120, 80, 130, 70])  # 50%+ swings
        assert self.cb._check_volatility_spike(prices)

    def test_multi_factor_trigger_logic(self):
        """Test multi-factor trigger logic with weighted scoring."""
        # Test individual factors
        factors = {
            'equity_drawdown': True,
            'consecutive_losses': False,
            'volatility_spike': False
        }

        score = self.cb._calculate_trigger_score(factors)
        expected_score = 0.6  # equity_drawdown weight
        assert abs(score - expected_score) < 0.01

        # Test multiple factors
        factors = {
            'equity_drawdown': True,
            'consecutive_losses': True,
            'volatility_spike': True
        }

        score = self.cb._calculate_trigger_score(factors)
        expected_score = 0.6 + 0.3 + 0.1  # All weights
        assert abs(score - expected_score) < 0.01

    def test_trigger_threshold_boundary_conditions(self):
        """Test trigger threshold boundary conditions."""
        # Test exact threshold values
        assert self.cb._check_equity_drawdown(10000, 9000)  # Exactly 10%
        assert not self.cb._check_equity_drawdown(10000, 9001)  # Just under 10%

        # Test floating point precision
        assert self.cb._check_equity_drawdown(10000, 8999.999)  # Very close to threshold
        assert not self.cb._check_equity_drawdown(10000, 9000.001)  # Just over threshold

    @pytest.mark.asyncio
    async def test_trigger_response_time(self):
        """Test trigger detection and response time (<100ms requirement)."""
        start_time = time.time()

        # Simulate trigger condition
        await self.cb.check_and_trigger({
            'equity': 8500,  # 15% drawdown from 10000
            'consecutive_losses': 6,
            'volatility': 0.08
        })

        response_time = (time.time() - start_time) * 1000  # Convert to ms
        assert response_time < 100, f"Response time {response_time}ms exceeds 100ms limit"
        assert self.cb.state == CircuitBreakerState.TRIGGERED


class TestCircuitBreakerStateManagement:
    """Test circuit breaker state management and transitions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig(
            equity_drawdown_threshold=0.1,
            consecutive_losses_threshold=3,
            cooling_period_minutes=1,  # Short for testing
            recovery_period_minutes=2
        )
        self.cb = CircuitBreaker(self.config)

    def test_state_transitions(self):
        """Test all state transitions in the state machine."""
        # Start in NORMAL
        assert self.cb.state == CircuitBreakerState.NORMAL

        # Trigger circuit breaker
        self.cb._trigger_circuit_breaker("Test trigger")
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Move to cooling
        self.cb._enter_cooling_period()
        assert self.cb.state == CircuitBreakerState.COOLING

        # Move to recovery
        self.cb._enter_recovery_period()
        assert self.cb.state == CircuitBreakerState.RECOVERY

        # Return to normal
        self.cb._return_to_normal()
        assert self.cb.state == CircuitBreakerState.NORMAL

    def test_manual_state_override(self):
        """Test manual state override functionality."""
        # Start in normal
        assert self.cb.state == CircuitBreakerState.NORMAL

        # Manually set to triggered
        self.cb.set_state(CircuitBreakerState.TRIGGERED, "Manual override")
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Manually reset to normal
        self.cb.reset_to_normal("Manual reset")
        assert self.cb.state == CircuitBreakerState.NORMAL

    def test_state_persistence(self):
        """Test state persistence across restarts."""
        # Set up circuit breaker with some state
        self.cb._trigger_circuit_breaker("Test trigger")
        self.cb._record_trade_result(100, True)  # Add some history

        # Simulate save
        state_data = self.cb.get_state_snapshot()

        # Create new instance and restore state
        new_cb = CircuitBreaker(self.config)
        new_cb.restore_state_snapshot(state_data)

        # Verify state was restored
        assert new_cb.state == self.cb.state
        assert len(new_cb.trigger_history) == len(self.cb.trigger_history)

    @pytest.mark.asyncio
    async def test_concurrent_state_access(self):
        """Test state machine integrity under concurrent access."""
        async def trigger_operations(cb, operation_id):
            """Simulate concurrent operations."""
            for i in range(10):
                if cb.state == CircuitBreakerState.NORMAL:
                    cb._trigger_circuit_breaker(f"Trigger {operation_id}-{i}")
                elif cb.state == CircuitBreakerState.TRIGGERED:
                    cb._enter_cooling_period()
                await asyncio.sleep(0.001)  # Small delay

        # Run multiple concurrent operations
        tasks = [
            trigger_operations(self.cb, i) for i in range(5)
        ]

        await asyncio.gather(*tasks)

        # State should be in a valid state (not corrupted)
        assert self.cb.state in CircuitBreakerState

        # History should be consistent
        assert len(self.cb.event_history) > 0


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with other components."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig()
        self.cb = CircuitBreaker(self.config)

        # Mock dependencies
        self.order_manager = MagicMock(spec=OrderManager)
        self.signal_router = MagicMock(spec=SignalRouter)
        self.risk_manager = MagicMock(spec=RiskManager)

    @pytest.mark.asyncio
    async def test_order_cancellation_integration(self):
        """Test order cancellation across exchange adapters during triggered state."""
        # Setup mocks
        self.order_manager.cancel_all_orders = AsyncMock(return_value=True)
        self.cb.order_manager = self.order_manager

        # Trigger circuit breaker
        await self.cb.check_and_trigger({'equity': 8000})  # 20% drawdown

        # Verify order cancellation was called
        self.order_manager.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_signal_blocking_integration(self):
        """Test signal blocking at strategy level when circuit breaker is active."""
        # Setup mocks
        self.signal_router.block_signals = AsyncMock()
        self.signal_router.unblock_signals = AsyncMock()
        self.cb.signal_router = self.signal_router

        # Trigger circuit breaker
        await self.cb.check_and_trigger({'consecutive_losses': 10})

        # Verify signal blocking was called
        self.signal_router.block_signals.assert_called_once()

        # Test unblocking after recovery
        self.cb._return_to_normal()
        self.signal_router.unblock_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_portfolio_freeze_integration(self):
        """Test portfolio freeze during cooling periods."""
        # Setup mocks
        self.risk_manager.freeze_portfolio = AsyncMock()
        self.risk_manager.unfreeze_portfolio = AsyncMock()
        self.cb.risk_manager = self.risk_manager

        # Trigger and move to cooling
        await self.cb.check_and_trigger({'volatility': 0.1})
        self.cb._enter_cooling_period()

        # Verify portfolio freeze was called
        self.risk_manager.freeze_portfolio.assert_called_once()

        # Test unfreezing after recovery
        self.cb._return_to_normal()
        self.risk_manager.unfreeze_portfolio.assert_called_once()

    @pytest.mark.asyncio
    async def test_anomaly_detector_integration(self):
        """Test coordination with AnomalyDetector."""
        # Setup mock anomaly detector
        anomaly_detector = MagicMock(spec=AnomalyDetector)
        anomaly_detector.detect_market_anomaly = AsyncMock(return_value=True)
        self.cb.anomaly_detector = anomaly_detector

        # Check integration
        market_data = {'prices': [100, 105, 95, 110, 90]}
        anomaly_detected = await self.cb._check_anomaly_integration(market_data)

        assert anomaly_detected
        anomaly_detector.detect_market_anomaly.assert_called_once_with(market_data)


class TestCircuitBreakerPerformance:
    """Test circuit breaker performance and reliability."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig()
        self.cb = CircuitBreaker(self.config)

    @pytest.mark.asyncio
    async def test_high_frequency_performance(self):
        """Test circuit breaker under high-frequency trading load."""
        # Simulate high-frequency trading scenario
        n_checks = 1000
        check_times = []

        for i in range(n_checks):
            start_time = time.time()

            # Perform check with varying conditions
            conditions = {
                'equity': 10000 + np.random.normal(0, 100),
                'consecutive_losses': np.random.poisson(2),
                'volatility': np.random.exponential(0.02)
            }

            await self.cb.check_and_trigger(conditions)
            check_times.append(time.time() - start_time)

        # Analyze performance
        avg_time = np.mean(check_times)
        max_time = np.max(check_times)
        p95_time = np.percentile(check_times, 95)

        # Performance requirements
        assert avg_time < 0.01, f"Average check time {avg_time:.4f}s exceeds 10ms limit"
        assert max_time < 0.1, f"Max check time {max_time:.4f}s exceeds 100ms limit"
        assert p95_time < 0.05, f"P95 check time {p95_time:.4f}s exceeds 50ms limit"

    @pytest.mark.asyncio
    async def test_memory_usage_during_monitoring(self):
        """Test memory usage during continuous monitoring."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Simulate continuous monitoring
        monitoring_duration = 30  # seconds
        check_interval = 0.1  # 100ms

        start_time = time.time()
        memory_samples = []

        while time.time() - start_time < monitoring_duration:
            # Perform monitoring check
            conditions = {
                'equity': 10000 + np.random.normal(0, 50),
                'consecutive_losses': 0,
                'volatility': 0.02
            }

            await self.cb.check_and_trigger(conditions)
            memory_samples.append(process.memory_info().rss)

            await asyncio.sleep(check_interval)

        # Analyze memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_samples)

        # Memory requirements (reasonable limits for monitoring)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increase {memory_increase/1024/1024:.1f}MB exceeds 50MB limit"
        assert max_memory < initial_memory * 1.1, "Memory usage increased by more than 10%"

    @pytest.mark.asyncio
    async def test_resource_exhaustion_behavior(self):
        """Test circuit breaker behavior during system resource exhaustion."""
        # Simulate memory pressure
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.percent = 95  # High memory usage

            # Circuit breaker should still function
            result = await self.cb.check_and_trigger({'equity': 9000})
            assert isinstance(result, bool)  # Should not crash

        # Simulate CPU pressure
        with patch('psutil.cpu_percent') as mock_cpu:
            mock_cpu.return_value = 95  # High CPU usage

            # Should still function
            result = await self.cb.check_and_trigger({'consecutive_losses': 6})
            assert isinstance(result, bool)


class TestCircuitBreakerEdgeCases:
    """Test circuit breaker edge cases and failure scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig()
        self.cb = CircuitBreaker(self.config)

    def test_market_data_gaps(self):
        """Test behavior during market data gaps and inconsistencies."""
        # Test with missing data
        conditions = {'equity': None, 'consecutive_losses': 3}
        result = self.cb._evaluate_triggers(conditions)
        assert isinstance(result, dict)  # Should handle gracefully

        # Test with invalid data types
        conditions = {'equity': 'invalid', 'consecutive_losses': [1, 2, 3]}
        result = self.cb._evaluate_triggers(conditions)
        assert isinstance(result, dict)  # Should handle gracefully

    def test_boundary_conditions(self):
        """Test threshold boundary conditions."""
        # Test exact boundary values
        assert self.cb._check_equity_drawdown(10000, 9000)  # Exactly 10%
        assert not self.cb._check_equity_drawdown(10000, 9000.01)  # Just over 10%

        # Test with zero values
        assert not self.cb._check_equity_drawdown(0, 0)  # Division by zero protection

        # Test negative values
        assert self.cb._check_equity_drawdown(10000, -1000)  # Negative equity

    @pytest.mark.asyncio
    async def test_partial_failure_recovery(self):
        """Test recovery from partial failures and corrupted state."""
        # Simulate corrupted trigger history
        self.cb.trigger_history = ["corrupted_data"]

        # Should handle gracefully
        result = await self.cb.check_and_trigger({'equity': 9000})
        assert isinstance(result, bool)

        # Simulate corrupted event history
        self.cb.event_history = [{"invalid": "data"}]

        # Should still function
        result = await self.cb.check_and_trigger({'consecutive_losses': 6})
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_dependent_service_unavailability(self):
        """Test graceful degradation when dependent services are unavailable."""
        # Setup mocks that raise exceptions
        self.cb.order_manager = MagicMock()
        self.cb.order_manager.cancel_all_orders = AsyncMock(side_effect=Exception("Service unavailable"))

        # Trigger circuit breaker
        result = await self.cb.check_and_trigger({'equity': 8000})

        # Should not crash, should log error and continue
        assert self.cb.state == CircuitBreakerState.TRIGGERED

    def test_extreme_market_conditions(self):
        """Test behavior under extreme market conditions."""
        # Test with extreme volatility
        prices = np.array([100, 1000, 1, 10000, 0.01])  # Extreme swings
        assert self.cb._check_volatility_spike(prices)

        # Test with extreme drawdown
        assert self.cb._check_equity_drawdown(10000, 1)  # 99.99% drawdown

        # Test with extreme consecutive losses
        for i in range(100):
            self.cb._record_trade_result(100, False)
        assert self.cb._check_consecutive_losses()


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration management."""

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Valid configuration
        config = CircuitBreakerConfig()
        assert config.equity_drawdown_threshold > 0
        assert config.consecutive_losses_threshold > 0

        # Invalid configuration should raise errors
        with pytest.raises(ValueError):
            CircuitBreakerConfig(equity_drawdown_threshold=-0.1)

        with pytest.raises(ValueError):
            CircuitBreakerConfig(consecutive_losses_threshold=0)

    def test_runtime_configuration_updates(self):
        """Test runtime configuration updates."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        # Update configuration
        new_config = CircuitBreakerConfig(equity_drawdown_threshold=0.05)  # Tighter threshold
        cb.update_config(new_config)

        # Verify update
        assert cb.config.equity_drawdown_threshold == 0.05

        # Test that it uses new configuration
        assert cb._check_equity_drawdown(10000, 9499)  # 5.01% drawdown should trigger


# Integration test fixtures
@pytest.fixture
def circuit_breaker():
    """Circuit breaker fixture."""
    config = CircuitBreakerConfig()
    return CircuitBreaker(config)


@pytest.fixture
def mock_order_manager():
    """Mock order manager fixture."""
    return MagicMock(spec=OrderManager)


@pytest.fixture
def mock_signal_router():
    """Mock signal router fixture."""
    return MagicMock(spec=SignalRouter)


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager fixture."""
    return MagicMock(spec=RiskManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
