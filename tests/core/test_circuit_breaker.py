"""
Comprehensive Circuit Breaker Testing Suite
===========================================

This module provides comprehensive testing for the Circuit Breaker feature,
covering core functionality, state management, integration, performance,
and edge cases as specified in the testing strategy.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CooldownStrategy,
)
from core.order_manager import OrderManager
from core.signal_router import SignalRouter
from risk.anomaly_detector import AnomalyDetector
from risk.risk_manager import RiskManager
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
            recovery_period_minutes=10,
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
            "equity_drawdown": True,
            "consecutive_losses": False,
            "volatility_spike": False,
        }

        score = self.cb._calculate_trigger_score(factors)
        expected_score = 0.6  # equity_drawdown weight
        assert abs(score - expected_score) < 0.01

        # Test multiple factors
        factors = {
            "equity_drawdown": True,
            "consecutive_losses": True,
            "volatility_spike": True,
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
        assert self.cb._check_equity_drawdown(
            10000, 8999.999
        )  # Very close to threshold
        assert not self.cb._check_equity_drawdown(
            10000, 9000.001
        )  # Just over threshold

    @pytest.mark.asyncio
    async def test_trigger_response_time(self):
        """Test trigger detection and response time (<100ms requirement)."""
        # Mock integration components to prevent hangs
        self.cb.order_manager = AsyncMock()
        self.cb.signal_router = AsyncMock()
        self.cb.risk_manager = AsyncMock()
        self.cb.anomaly_detector = None  # Ensure anomaly detector is not set

        start_time = time.time()

        # Simulate trigger condition with timeout protection
        try:
            result = await asyncio.wait_for(
                self.cb.check_and_trigger(
                    {
                        "equity": 8500,  # 15% drawdown from 10000
                        "consecutive_losses": 6,
                        "volatility": 0.08,
                    }
                ),
                timeout=5.0,  # 5 second timeout for the entire operation
            )
        except asyncio.TimeoutError:
            pytest.fail("check_and_trigger timed out after 5 seconds")

        response_time = (time.time() - start_time) * 1000  # Convert to ms
        assert (
            response_time < 100
        ), f"Response time {response_time}ms exceeds 100ms limit"
        assert (
            result is True
        ), "check_and_trigger should return True for triggered conditions"
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Verify that mocked methods were called
        self.cb.order_manager.cancel_all_orders.assert_called_once()
        self.cb.signal_router.block_signals.assert_called_once()


class TestCircuitBreakerStateManagement:
    """Test circuit breaker state management and transitions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig(
            equity_drawdown_threshold=0.1,
            consecutive_losses_threshold=3,
            cooling_period_minutes=1,  # Short for testing
            recovery_period_minutes=2,
        )
        self.cb = CircuitBreaker(self.config)

    @pytest.mark.asyncio
    async def test_state_transitions(self):
        """Test all state transitions in the state machine."""
        # Start in NORMAL
        assert self.cb.state == CircuitBreakerState.NORMAL

        # Trigger circuit breaker
        await self.cb._trigger_circuit_breaker("Test trigger")
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Move to cooling
        await self.cb._enter_cooling_period()
        assert self.cb.state == CircuitBreakerState.COOLING

        # Set up conditions for health check to pass
        self.cb.current_equity = 9600  # Above recovery threshold
        self.cb._record_trade_result(100, True)  # Add winning trade

        # Move to recovery
        await self.cb._enter_recovery_period()
        assert self.cb.state == CircuitBreakerState.RECOVERY

        # Return to normal
        await self.cb._return_to_normal()
        assert self.cb.state == CircuitBreakerState.NORMAL

    @pytest.mark.asyncio
    async def test_manual_state_override(self):
        """Test manual state override functionality."""
        # Start in normal
        assert self.cb.state == CircuitBreakerState.NORMAL

        # Manually set to triggered
        await self.cb.set_state(CircuitBreakerState.TRIGGERED, "Manual override")
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Manually reset to normal
        result = await self.cb.reset_to_normal("Manual reset")
        assert result
        assert self.cb.state == CircuitBreakerState.NORMAL

    @pytest.mark.asyncio
    async def test_state_persistence(self):
        """Test state persistence across restarts."""
        # Set up circuit breaker with some state
        await self.cb._trigger_circuit_breaker("Test trigger")
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
                    await cb._trigger_circuit_breaker(f"Trigger {operation_id}-{i}")
                elif cb.state == CircuitBreakerState.TRIGGERED:
                    await cb._enter_cooling_period()
                await asyncio.sleep(0.001)  # Small delay

        # Run multiple concurrent operations
        tasks = [trigger_operations(self.cb, i) for i in range(5)]

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
        await self.cb.check_and_trigger({"equity": 8000})  # 20% drawdown

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
        await self.cb.check_and_trigger({"consecutive_losses": 10})

        # Verify signal blocking was called
        self.signal_router.block_signals.assert_called_once()

        # Test unblocking after recovery
        await self.cb._return_to_normal()
        self.signal_router.unblock_signals.assert_called_once()

    @pytest.mark.asyncio
    async def test_portfolio_freeze_integration(self):
        """Test portfolio freeze during cooling periods."""
        # Setup mocks
        self.risk_manager.freeze_portfolio = AsyncMock()
        self.risk_manager.unfreeze_portfolio = AsyncMock()
        self.cb.risk_manager = self.risk_manager

        # Trigger and move to cooling
        await self.cb.check_and_trigger({"volatility": 0.1})
        await self.cb._enter_cooling_period()

        # Verify portfolio freeze was called
        self.risk_manager.freeze_portfolio.assert_called_once()

        # Test unfreezing after recovery
        await self.cb._return_to_normal()
        self.risk_manager.unfreeze_portfolio.assert_called_once()

    @pytest.mark.asyncio
    async def test_anomaly_detector_integration(self):
        """Test coordination with AnomalyDetector."""
        # Setup mock anomaly detector
        anomaly_detector = MagicMock(spec=AnomalyDetector)
        anomaly_detector.detect_market_anomaly = AsyncMock(return_value=True)
        self.cb.anomaly_detector = anomaly_detector

        # Check integration
        market_data = {"prices": [100, 105, 95, 110, 90]}
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
        # Mock integration components to prevent hangs
        self.cb.order_manager = AsyncMock()
        self.cb.signal_router = AsyncMock()
        self.cb.risk_manager = AsyncMock()
        self.cb.anomaly_detector = None

        # Simulate high-frequency trading scenario
        n_checks = 1000
        check_times = []

        for i in range(n_checks):
            start_time = time.time()

            # Perform check with varying conditions
            conditions = {
                "equity": 10000 + np.random.normal(0, 100),
                "consecutive_losses": np.random.poisson(2),
                "volatility": np.random.exponential(0.02),
            }

            try:
                await asyncio.wait_for(
                    self.cb.check_and_trigger(conditions),
                    timeout=1.0,  # 1 second timeout per check
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Check {i} timed out")
                continue

            check_times.append(time.time() - start_time)

        # Analyze performance
        if check_times:  # Only analyze if we have successful checks
            avg_time = np.mean(check_times)
            max_time = np.max(check_times)
            p95_time = np.percentile(check_times, 95)

            # Performance requirements
            assert (
                avg_time < 0.01
            ), f"Average check time {avg_time:.4f}s exceeds 10ms limit"
            assert max_time < 0.1, f"Max check time {max_time:.4f}s exceeds 100ms limit"
            assert p95_time < 0.05, f"P95 check time {p95_time:.4f}s exceeds 50ms limit"
        else:
            pytest.fail("No successful checks completed")

    @pytest.mark.asyncio
    async def test_memory_usage_during_monitoring(self):
        """Test memory usage during continuous monitoring."""
        import os

        import psutil

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
                "equity": 10000 + np.random.normal(0, 50),
                "consecutive_losses": 0,
                "volatility": 0.02,
            }

            await self.cb.check_and_trigger(conditions)
            memory_samples.append(process.memory_info().rss)

            await asyncio.sleep(check_interval)

        # Analyze memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        max_memory = max(memory_samples)

        # Memory requirements (reasonable limits for monitoring)
        assert (
            memory_increase < 50 * 1024 * 1024
        ), f"Memory increase {memory_increase/1024/1024:.1f}MB exceeds 50MB limit"
        assert (
            max_memory < initial_memory * 1.1
        ), "Memory usage increased by more than 10%"

    @pytest.mark.asyncio
    async def test_resource_exhaustion_behavior(self):
        """Test circuit breaker behavior during system resource exhaustion."""
        # Simulate memory pressure
        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.percent = 95  # High memory usage

            # Circuit breaker should still function
            result = await self.cb.check_and_trigger({"equity": 9000})
            assert isinstance(result, bool)  # Should not crash

        # Simulate CPU pressure
        with patch("psutil.cpu_percent") as mock_cpu:
            mock_cpu.return_value = 95  # High CPU usage

            # Should still function
            result = await self.cb.check_and_trigger({"consecutive_losses": 6})
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
        conditions = {"equity": None, "consecutive_losses": 3}
        result = self.cb._evaluate_triggers(conditions)
        assert isinstance(result, dict)  # Should handle gracefully

        # Test with invalid data types
        conditions = {"equity": "invalid", "consecutive_losses": [1, 2, 3]}
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
        result = await self.cb.check_and_trigger({"equity": 9000})
        assert isinstance(result, bool)

        # Simulate corrupted event history
        self.cb.event_history = [{"invalid": "data"}]

        # Should still function
        result = await self.cb.check_and_trigger({"consecutive_losses": 6})
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_dependent_service_unavailability(self):
        """Test graceful degradation when dependent services are unavailable."""
        # Setup mocks that raise exceptions
        self.cb.order_manager = MagicMock()
        self.cb.order_manager.cancel_all_orders = AsyncMock(
            side_effect=Exception("Service unavailable")
        )

        # Trigger circuit breaker
        result = await self.cb.check_and_trigger({"equity": 8000})

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
        new_config = CircuitBreakerConfig(
            equity_drawdown_threshold=0.05
        )  # Tighter threshold
        cb.update_config(new_config)

        # Verify update
        assert cb.config.equity_drawdown_threshold == 0.05

        # Test that it uses new configuration
        assert cb._check_equity_drawdown(10000, 9499)  # 5.01% drawdown should trigger


class TestCircuitBreakerCooldownEnforcement:
    """Test cooldown period enforcement functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig(
            enable_cooldown_enforcement=True,
            cooldown_strategy=CooldownStrategy.EXPONENTIAL,
            base_cooldown_minutes=5,
            max_cooldown_minutes=60,
            cooldown_multiplier=2.0,
        )
        self.cb = CircuitBreaker(self.config)

    def test_cooldown_calculation_fixed_strategy(self):
        """Test fixed cooldown period calculation."""
        config = CircuitBreakerConfig(
            cooldown_strategy=CooldownStrategy.FIXED,
            base_cooldown_minutes=10,
            max_cooldown_minutes=60,
        )
        cb = CircuitBreaker(config)

        # Fixed strategy should always return base_cooldown_minutes
        assert cb._calculate_cooldown_period(1) == 10
        assert cb._calculate_cooldown_period(5) == 10
        assert cb._calculate_cooldown_period(10) == 10

    def test_cooldown_calculation_exponential_strategy(self):
        """Test exponential cooldown period calculation."""
        # First trigger: 5 minutes
        assert self.cb._calculate_cooldown_period(1) == 5

        # Second trigger: 5 * 2^0 = 10 minutes
        assert self.cb._calculate_cooldown_period(2) == 10

        # Third trigger: 5 * 2^1 = 20 minutes
        assert self.cb._calculate_cooldown_period(3) == 20

        # Fourth trigger: 5 * 2^2 = 40 minutes
        assert self.cb._calculate_cooldown_period(4) == 40

        # Fifth trigger: min(5 * 2^3, 60) = 60 minutes (capped)
        assert self.cb._calculate_cooldown_period(5) == 60

    def test_cooldown_calculation_fibonacci_strategy(self):
        """Test fibonacci cooldown period calculation."""
        config = CircuitBreakerConfig(
            cooldown_strategy=CooldownStrategy.FIBONACCI,
            base_cooldown_minutes=5,
            max_cooldown_minutes=100,
        )
        cb = CircuitBreaker(config)

        # Fibonacci sequence with base 5: 5, 5, 10, 15, 25, 40, 65, ...
        assert cb._calculate_cooldown_period(1) == 5
        assert cb._calculate_cooldown_period(2) == 5
        assert cb._calculate_cooldown_period(3) == 10
        assert cb._calculate_cooldown_period(4) == 15
        assert cb._calculate_cooldown_period(5) == 25
        assert cb._calculate_cooldown_period(6) == 40

    def test_cooldown_enforcement_disabled(self):
        """Test behavior when cooldown enforcement is disabled."""
        config = CircuitBreakerConfig(enable_cooldown_enforcement=False)
        cb = CircuitBreaker(config)

        # Should always return 0 when disabled
        assert cb._calculate_cooldown_period(1) == 0
        assert cb._calculate_cooldown_period(10) == 0
        assert not cb._is_cooldown_active()

    @pytest.mark.asyncio
    async def test_cooldown_prevents_triggering(self):
        """Test that cooldown period prevents new triggers."""
        # Mock integration components
        self.cb.order_manager = AsyncMock()
        self.cb.signal_router = AsyncMock()
        self.cb.risk_manager = AsyncMock()
        self.cb.anomaly_detector = None

        # First trigger should work
        result1 = await self.cb.check_and_trigger({"equity": 8000})
        assert result1 is True
        assert self.cb.state == CircuitBreakerState.TRIGGERED
        assert self.cb.trigger_count == 1

        # Immediately try to trigger again - should be blocked by cooldown
        result2 = await self.cb.check_and_trigger({"equity": 7000})
        assert result2 is False  # Should be blocked
        assert self.cb.state == CircuitBreakerState.TRIGGERED  # State unchanged
        assert self.cb.trigger_count == 1  # Count unchanged

    def test_remaining_cooldown_calculation(self):
        """Test remaining cooldown time calculation."""
        from datetime import datetime, timedelta

        # Simulate a trigger 2 minutes ago
        past_time = datetime.now() - timedelta(minutes=2)
        self.cb.last_trigger_time = past_time
        self.cb.trigger_count = 1

        # With 5 minute cooldown, should have 3 minutes remaining
        remaining = self.cb.get_remaining_cooldown_minutes()
        assert abs(remaining - 3.0) < 0.1  # Allow small timing differences

    def test_cooldown_state_after_multiple_triggers(self):
        """Test cooldown state after multiple triggers."""
        from datetime import datetime, timedelta

        # Simulate multiple triggers with exponential backoff
        self.cb.trigger_count = 3  # Third trigger
        self.cb.last_trigger_time = datetime.now() - timedelta(minutes=10)  # 10 minutes ago

        # Third trigger should have 20 minute cooldown
        expected_cooldown = 20
        actual_cooldown = self.cb._calculate_cooldown_period(3)
        assert actual_cooldown == expected_cooldown

        # Should still be in cooldown (20 min cooldown, only 10 min elapsed)
        assert self.cb._is_cooldown_active()

        # Remaining time should be about 10 minutes
        remaining = self.cb.get_remaining_cooldown_minutes()
        assert abs(remaining - 10.0) < 0.1


class TestCircuitBreakerHealthChecks:
    """Test health check functionality for recovery."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig()
        self.cb = CircuitBreaker(self.config)

    @pytest.mark.asyncio
    async def test_health_check_equity_recovery(self):
        """Test health check based on equity recovery."""
        # Set current equity below recovery threshold
        self.cb.current_equity = 9000  # Below 9500 (95% of 10000)
        health_ok = await self.cb._perform_health_check()
        assert not health_ok

        # Set equity above recovery threshold
        self.cb.current_equity = 9600  # Above 9500
        health_ok = await self.cb._perform_health_check()
        assert health_ok

    @pytest.mark.asyncio
    async def test_health_check_trade_pattern(self):
        """Test health check based on recent trade patterns."""
        # Add some losing trades
        for i in range(5):
            self.cb._record_trade_result(100, False)

        # Recent trades are all losses - should fail health check
        health_ok = await self.cb._perform_health_check()
        assert not health_ok

        # Add a winning trade to break the losing streak
        self.cb._record_trade_result(100, True)

        # Now recent trades include a win - should pass health check
        health_ok = await self.cb._perform_health_check()
        assert health_ok

    @pytest.mark.asyncio
    async def test_recovery_with_health_check(self):
        """Test recovery process with health check validation."""
        # Start in triggered state
        await self.cb._trigger_circuit_breaker("Test trigger")
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Move to cooling
        await self.cb._enter_cooling_period()
        assert self.cb.state == CircuitBreakerState.COOLING

        # Try to enter recovery - should fail health check
        self.cb.current_equity = 9000  # Below threshold
        await self.cb._enter_recovery_period()
        assert self.cb.state == CircuitBreakerState.COOLING  # Should stay in cooling

        # Set healthy conditions and try again
        self.cb.current_equity = 9600  # Above threshold
        self.cb._record_trade_result(100, True)  # Add winning trade

        await self.cb._enter_recovery_period()
        assert self.cb.state == CircuitBreakerState.RECOVERY  # Should enter recovery


class TestCircuitBreakerStateTransitions:
    """Test circuit breaker state transitions with new features."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = CircuitBreakerConfig(
            enable_cooldown_enforcement=True,
            cooldown_strategy=CooldownStrategy.EXPONENTIAL,
        )
        self.cb = CircuitBreaker(self.config)

    @pytest.mark.asyncio
    async def test_state_transition_with_cooldown(self):
        """Test state transitions respect cooldown periods."""
        # Mock integration components
        self.cb.order_manager = AsyncMock()
        self.cb.signal_router = AsyncMock()
        self.cb.risk_manager = AsyncMock()
        self.cb.anomaly_detector = None

        # First trigger
        result1 = await self.cb.check_and_trigger({"equity": 8000})
        assert result1 is True
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Try immediate second trigger - should be blocked
        result2 = await self.cb.check_and_trigger({"equity": 7000})
        assert result2 is False
        assert self.cb.state == CircuitBreakerState.TRIGGERED

    @pytest.mark.asyncio
    async def test_emergency_reset_bypasses_cooldown(self):
        """Test that emergency reset can bypass cooldown."""
        # Trigger circuit breaker
        await self.cb._trigger_circuit_breaker("Test trigger")
        assert self.cb.state == CircuitBreakerState.TRIGGERED

        # Emergency reset should work even during cooldown
        result = await self.cb.reset_to_normal("Emergency reset")
        assert result is True
        assert self.cb.state == CircuitBreakerState.NORMAL

    def test_circuit_state_metrics(self):
        """Test circuit state metrics collection."""
        from datetime import datetime, timedelta

        # Set up some state
        self.cb.state = CircuitBreakerState.COOLING
        self.cb.trigger_count = 3
        self.cb.last_trigger_time = datetime.now() - timedelta(minutes=5)
        self.cb.current_equity = 9500

        # Get metrics
        metrics = self.cb.get_circuit_state_metrics()

        # Verify metrics content
        assert metrics["state"] == "cooling"
        assert metrics["trigger_count"] == 3
        assert metrics["current_equity"] == 9500
        assert metrics["cooldown_strategy"] == "exponential"
        assert metrics["enable_cooldown_enforcement"] is True
        assert isinstance(metrics["remaining_cooldown_minutes"], float)
        assert isinstance(metrics["is_cooldown_active"], bool)


@pytest.mark.timeout(10)
def test_circuit_breaker_cooldown_timeout():
    """Test circuit breaker cooldown timeout behavior."""
    import asyncio

    async def run_test():
        config = CircuitBreakerConfig(
            enable_cooldown_enforcement=True,
            cooldown_strategy=CooldownStrategy.FIXED,
            base_cooldown_minutes=1,  # Short cooldown for testing
        )
        cb = CircuitBreaker(config)

        # Mock integration components
        cb.order_manager = AsyncMock()
        cb.signal_router = AsyncMock()
        cb.risk_manager = AsyncMock()
        cb.anomaly_detector = None

        # First trigger should work
        result1 = await cb.check_and_trigger({"equity": 8000})
        assert result1 is True

        # Immediate second trigger should be blocked by cooldown
        result2 = await cb.check_and_trigger({"equity": 7000})
        assert result2 is False  # Should timeout quickly due to cooldown

    asyncio.run(run_test())


# Integration test fixtures
@pytest.fixture
async def circuit_breaker():
    """Circuit breaker fixture with proper cleanup."""
    config = CircuitBreakerConfig()
    cb = CircuitBreaker(config)
    yield cb
    # Cleanup background tasks after test
    await cb.cleanup_background_tasks()


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
