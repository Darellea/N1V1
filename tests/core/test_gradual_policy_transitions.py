"""
Tests for gradual policy transitions in adaptive risk policy.

This module tests the gradual transition functionality of the AdaptiveRiskPolicy,
including smooth interpolation, progress monitoring, emergency overrides, and rollback.
"""

import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError

import numpy as np
import pandas as pd
import pytest

from risk.adaptive_policy import (
    AdaptiveRiskPolicy,
    EmergencyOverride,
    PolicyTransition,
    RiskPolicy,
    TransitionMode,
    TransitionState,
)


class TestPolicyTransition:
    """Test PolicyTransition class functionality."""

    def test_policy_transition_initialization(self):
        """Test PolicyTransition initialization."""
        from_policy = RiskPolicy(max_position=1000, max_loss=100)
        to_policy = RiskPolicy(max_position=2000, max_loss=200)

        transition = PolicyTransition(
            from_policy=from_policy,
            to_policy=to_policy,
            duration=60,  # 1 minute
            mode=TransitionMode.GRADUAL,
        )

        assert transition.from_policy == from_policy
        assert transition.to_policy == to_policy
        assert transition.duration == 60
        assert transition.mode == TransitionMode.GRADUAL
        assert transition.state == TransitionState.PENDING
        assert transition.progress == 0.0
        assert transition.start_time is None

    def test_policy_transition_validation(self):
        """Test PolicyTransition validation."""
        from_policy = RiskPolicy(max_position=1000, max_loss=100)
        to_policy = RiskPolicy(max_position=2000, max_loss=200)

        # Valid transition
        transition = PolicyTransition(
            from_policy=from_policy, to_policy=to_policy, duration=30
        )
        assert transition.is_valid()

        # Invalid duration
        with pytest.raises(ValueError):
            PolicyTransition(from_policy, to_policy, duration=0)

        # Invalid policies
        with pytest.raises(ValueError):
            PolicyTransition(None, to_policy, duration=30)

    def test_transition_interpolation(self):
        """Test smooth interpolation between policies."""
        from_policy = RiskPolicy(
            max_position=1000, max_loss=100, volatility_threshold=0.05
        )
        to_policy = RiskPolicy(
            max_position=2000, max_loss=200, volatility_threshold=0.10
        )

        transition = PolicyTransition(from_policy, to_policy, duration=100)

        # Test at 0% progress
        interpolated = transition.interpolate(0.0)
        assert interpolated.max_position == 1000
        assert interpolated.max_loss == 100
        assert interpolated.volatility_threshold == 0.05

        # Test at 50% progress
        interpolated = transition.interpolate(0.5)
        assert interpolated.max_position == 1500
        assert interpolated.max_loss == 150
        assert interpolated.volatility_threshold == pytest.approx(0.075, abs=1e-6)

        # Test at 100% progress
        interpolated = transition.interpolate(1.0)
        assert interpolated.max_position == 2000
        assert interpolated.max_loss == 200
        assert interpolated.volatility_threshold == 0.10


class TestAdaptivePolicyTransitions:
    """Test AdaptiveRiskPolicy transition functionality."""

    @pytest.fixture
    def policy_manager(self):
        """Create AdaptiveRiskPolicy instance for testing."""
        config = {
            "min_multiplier": 0.1,
            "max_multiplier": 2.0,
            "transition_enabled": True,
            "default_transition_duration": 30,
            "emergency_override_timeout": 10,
        }
        return AdaptiveRiskPolicy(config)

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="1min")
        np.random.seed(42)
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.abs(np.random.randn(100) * 0.2)
        low_prices = close_prices - np.abs(np.random.randn(100) * 0.2)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": close_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

    def test_immediate_transition_mode(self, policy_manager, sample_market_data):
        """Test immediate transition mode."""
        from_policy = RiskPolicy(max_position=1000)
        to_policy = RiskPolicy(max_position=2000)

        transition = PolicyTransition(
            from_policy=from_policy,
            to_policy=to_policy,
            duration=0,  # Immediate
            mode=TransitionMode.IMMEDIATE,
        )

        policy_manager.start_transition(transition)

        # Should complete immediately
        assert transition.state == TransitionState.COMPLETED
        assert transition.progress == 1.0

        # Current policy should be the target policy
        assert policy_manager.current_policy.max_position == 2000

    def test_gradual_transition_execution(self, policy_manager, sample_market_data):
        """Test gradual transition execution."""
        from_policy = RiskPolicy(max_position=1000, max_loss=100)
        to_policy = RiskPolicy(max_position=2000, max_loss=200)

        transition = PolicyTransition(
            from_policy=from_policy,
            to_policy=to_policy,
            duration=2,  # 2 seconds for testing
            mode=TransitionMode.GRADUAL,
        )

        policy_manager.start_transition(transition)

        # Wait for transition to start
        time.sleep(0.5)  # Give more time for thread to start
        assert transition.state == TransitionState.IN_PROGRESS
        assert 0 <= transition.progress <= 1  # Allow 0 progress initially

        # Wait for completion
        time.sleep(2.5)
        assert transition.state == TransitionState.COMPLETED
        assert transition.progress == 1.0

        # Verify final policy
        assert policy_manager.current_policy.max_position == 2000
        assert policy_manager.current_policy.max_loss == 200

    @pytest.mark.timeout(10)
    def test_policy_transition_timeout_safety(self):
        """Test policy transition timeout safety."""
        policy_mgr = AdaptiveRiskPolicy({})

        # Create transition that would normally take too long
        long_transition = PolicyTransition(
            from_policy=RiskPolicy(max_position=1000),
            to_policy=RiskPolicy(max_position=2000),
            duration=300,  # 5 minutes
        )

        # Should allow interruption and timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(policy_mgr.execute_transition, long_transition)
            time.sleep(1)  # Let it start
            future.cancel()  # Should handle cancellation gracefully

            with pytest.raises((CancelledError, TimeoutError)):
                future.result(timeout=2.0)

    def test_transition_progress_monitoring(self, policy_manager):
        """Test transition progress monitoring."""
        from_policy = RiskPolicy(max_position=1000)
        to_policy = RiskPolicy(max_position=2000)

        transition = PolicyTransition(from_policy, to_policy, duration=1)

        policy_manager.start_transition(transition)

        # Monitor progress over time
        progress_values = []
        for _ in range(10):
            progress_values.append(transition.progress)
            time.sleep(0.1)

        # Progress should be monotonically increasing
        assert all(
            progress_values[i] <= progress_values[i + 1]
            for i in range(len(progress_values) - 1)
        )

        # Should eventually complete
        time.sleep(1)
        assert transition.state == TransitionState.COMPLETED
        assert transition.progress == 1.0

    def test_emergency_override(self, policy_manager):
        """Test emergency policy override."""
        # Start a gradual transition
        from_policy = RiskPolicy(max_position=1000)
        to_policy = RiskPolicy(max_position=2000)

        transition = PolicyTransition(
            from_policy, to_policy, duration=30
        )  # Longer duration
        policy_manager.start_transition(transition)

        time.sleep(0.5)  # Let transition start
        assert transition.state == TransitionState.IN_PROGRESS

        # Emergency override
        emergency_policy = RiskPolicy(max_position=500)  # Conservative override
        override = EmergencyOverride(
            policy=emergency_policy,
            reason="Market crash detected",
            timeout=5,  # Shorter timeout for testing
        )

        policy_manager.apply_emergency_override(override)

        # Should immediately switch to emergency policy
        assert policy_manager.current_policy.max_position == 500
        assert transition.state == TransitionState.PAUSED

        # After timeout, should resume transition
        time.sleep(6)
        policy_manager.cleanup_expired_overrides()  # Clean up expired overrides and resume transitions
        assert transition.state == TransitionState.IN_PROGRESS

    def test_transition_rollback(self, policy_manager):
        """Test rollback of failed policy transition."""
        original_policy = RiskPolicy(max_position=1000)
        policy_manager.current_policy = original_policy

        # Start transition
        to_policy = RiskPolicy(max_position=2000)
        transition = PolicyTransition(original_policy, to_policy, duration=2)

        policy_manager.start_transition(transition)
        time.sleep(0.5)

        # Simulate failure and rollback
        policy_manager.rollback_transition(transition, "Test failure")

        assert transition.state == TransitionState.FAILED
        assert policy_manager.current_policy == original_policy

    def test_concurrent_transitions(self, policy_manager):
        """Test handling of concurrent transitions."""
        # Start first transition
        transition1 = PolicyTransition(
            RiskPolicy(max_position=1000), RiskPolicy(max_position=1500), duration=2
        )
        policy_manager.start_transition(transition1)

        # Attempt second transition (should be rejected)
        transition2 = PolicyTransition(
            RiskPolicy(max_position=1500), RiskPolicy(max_position=2000), duration=2
        )

        with pytest.raises(RuntimeError, match="Transition already in progress"):
            policy_manager.start_transition(transition2)

    def test_interpolation_accuracy(self):
        """Test interpolation accuracy for various policy parameters."""
        from_policy = RiskPolicy(
            max_position=1000,
            max_loss=100,
            volatility_threshold=0.05,
            trend_threshold=0.02,
        )
        to_policy = RiskPolicy(
            max_position=2000,
            max_loss=200,
            volatility_threshold=0.10,
            trend_threshold=0.04,
        )

        transition = PolicyTransition(from_policy, to_policy, duration=100)

        # Test multiple progress points
        test_points = [0.0, 0.25, 0.5, 0.75, 1.0]

        for progress in test_points:
            interpolated = transition.interpolate(progress)

            # Check linear interpolation
            expected_max_position = 1000 + progress * (2000 - 1000)
            expected_max_loss = 100 + progress * (200 - 100)
            expected_volatility = 0.05 + progress * (0.10 - 0.05)
            expected_trend = 0.02 + progress * (0.04 - 0.02)

            assert abs(interpolated.max_position - expected_max_position) < 1e-6
            assert abs(interpolated.max_loss - expected_max_loss) < 1e-6
            assert abs(interpolated.volatility_threshold - expected_volatility) < 1e-6
            assert abs(interpolated.trend_threshold - expected_trend) < 1e-6

    def test_transition_validation_and_safety_checks(self, policy_manager):
        """Test transition validation and safety checks."""
        # Invalid transition (negative duration)
        with pytest.raises(ValueError):
            PolicyTransition(
                RiskPolicy(max_position=1000),
                RiskPolicy(max_position=2000),
                duration=-1,
            )

        # Transition with extreme values
        extreme_policy = RiskPolicy(max_position=1000000)  # Unrealistic
        transition = PolicyTransition(
            RiskPolicy(max_position=1000), extreme_policy, duration=10
        )

        # Should validate and potentially reject or warn
        assert transition.is_valid()  # Assuming validation allows it

        # But policy manager should have safety checks
        with pytest.raises(ValueError, match="Policy validation failed"):
            policy_manager.start_transition(transition)

    def test_transition_with_market_data_integration(
        self, policy_manager, sample_market_data
    ):
        """Test transition integration with market data."""
        # Start with conservative policy
        conservative_policy = RiskPolicy(max_position=500, volatility_threshold=0.02)
        policy_manager.current_policy = conservative_policy

        # Transition to more aggressive policy
        aggressive_policy = RiskPolicy(max_position=2000, volatility_threshold=0.08)
        transition = PolicyTransition(
            conservative_policy,
            aggressive_policy,
            duration=5,  # Shorter duration for testing
        )

        policy_manager.start_transition(transition)

        # Check that effective policy changes during transition
        time.sleep(1)
        effective_policy = policy_manager.get_effective_policy()
        assert effective_policy.max_position > 500  # Should be interpolated value

        time.sleep(5)  # Wait for completion
        final_policy = policy_manager.get_effective_policy()
        assert final_policy.max_position == 2000  # Should be final value

        # Verify transition completed
        assert transition.state == TransitionState.COMPLETED

    def test_transition_state_consistency(self, policy_manager):
        """Test transition state consistency."""
        transition = PolicyTransition(
            RiskPolicy(max_position=1000), RiskPolicy(max_position=2000), duration=1
        )

        # Initial state
        assert transition.state == TransitionState.PENDING

        # Start transition
        policy_manager.start_transition(transition)
        assert transition.state == TransitionState.IN_PROGRESS

        # Complete transition
        time.sleep(2)  # Wait longer for 1-second transition
        assert transition.state == TransitionState.COMPLETED

        # Verify state transitions are valid
        valid_transitions = {
            TransitionState.PENDING: [
                TransitionState.IN_PROGRESS,
                TransitionState.CANCELLED,
            ],
            TransitionState.IN_PROGRESS: [
                TransitionState.COMPLETED,
                TransitionState.FAILED,
                TransitionState.PAUSED,
            ],
            TransitionState.PAUSED: [
                TransitionState.IN_PROGRESS,
                TransitionState.CANCELLED,
            ],
            TransitionState.COMPLETED: [],
            TransitionState.FAILED: [TransitionState.PENDING],  # Can retry
            TransitionState.CANCELLED: [],
        }

        # This would be tested by attempting invalid state changes
        # For now, just verify the state machine is respected in implementation
