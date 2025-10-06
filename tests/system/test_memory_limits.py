"""
Memory limits and hard memory management tests.

Tests for hard memory limits, graceful degradation, emergency cleanup,
and memory usage forecasting functionality.
"""

import gc
import os
import time
import pytest
import psutil
from unittest.mock import patch, MagicMock

from core.memory_manager import MemoryManager, get_memory_manager


class TestMemoryLimits:
    """Test memory limit enforcement and management."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["TESTING"] = "1"  # Disable monitoring thread
        self.memory_manager = MemoryManager(enable_monitoring=False)

    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.shutdown()
        if "TESTING" in os.environ:
            del os.environ["TESTING"]

    @pytest.mark.timeout(15)
    def test_hard_limit_enforcement(self):
        """Test that hard memory limits are enforced."""
        # Set low hard limit
        self.memory_manager.set_memory_thresholds(hard_limit_mb=50.0)

        # Should allow allocation under limit
        assert self.memory_manager.allocate_memory("test_component", 20.0)

        # Should deny allocation that would exceed limit
        assert not self.memory_manager.allocate_memory("test_component", 40.0)

        # Check usage tracking
        usage = self.memory_manager.get_component_memory_usage()
        assert usage["test_component"] == 20.0

    @pytest.mark.timeout(15)
    def test_component_specific_limits(self):
        """Test component-specific memory limits."""
        # Set component limits
        self.memory_manager._component_limits["cache"] = 10.0
        self.memory_manager._component_limits["ml_models"] = 50.0

        # Should allow within component limit
        assert self.memory_manager.allocate_memory("cache", 5.0)

        # Should deny exceeding component limit
        assert not self.memory_manager.allocate_memory("cache", 8.0)

        # Different component should have different limit
        assert self.memory_manager.allocate_memory("ml_models", 40.0)

    @pytest.mark.timeout(15)
    def test_memory_allocation_tracking(self):
        """Test memory allocation and deallocation tracking."""
        # Allocate memory
        assert self.memory_manager.allocate_memory("component1", 10.0)
        assert self.memory_manager.allocate_memory("component2", 15.0)

        usage = self.memory_manager.get_component_memory_usage()
        assert usage["component1"] == 10.0
        assert usage["component2"] == 15.0

        # Deallocate memory
        self.memory_manager.deallocate_memory("component1", 5.0)
        usage = self.memory_manager.get_component_memory_usage()
        assert usage["component1"] == 5.0

    @pytest.mark.timeout(15)
    def test_graceful_degradation_activation(self):
        """Test graceful degradation is triggered at threshold."""
        self.memory_manager.set_memory_thresholds(graceful_degradation_mb=100.0)

        # Mock memory usage to trigger degradation
        with patch.object(self.memory_manager, 'get_memory_usage', return_value=120.0):
            self.memory_manager._check_memory_usage()

            # Should have started degradation
            assert self.memory_manager._degradation_active

    @pytest.mark.timeout(15)
    def test_graceful_degradation_steps(self):
        """Test graceful degradation executes steps in order."""
        self.memory_manager.start_graceful_degradation()

        # Should execute first step
        assert self.memory_manager._current_degradation_level == 1

        # Check that degradation steps are defined
        assert len(self.memory_manager._degradation_steps) > 0

    @pytest.mark.timeout(15)
    def test_emergency_mode_detection(self):
        """Test emergency mode is properly detected."""
        assert not self.memory_manager.is_in_emergency_mode()

        # Set emergency threshold low and trigger
        self.memory_manager.set_memory_thresholds(emergency_cleanup_mb=50.0)

        with patch.object(self.memory_manager, 'get_memory_usage', return_value=60.0):
            self.memory_manager._check_memory_usage()

            assert self.memory_manager.is_in_emergency_mode()

    @pytest.mark.timeout(15)
    def test_memory_forecasting(self):
        """Test memory usage forecasting functionality."""
        # Add some mock snapshots
        current_time = time.time()
        snapshots = [
            {"timestamp": current_time - 50, "memory_mb": 100.0},
            {"timestamp": current_time - 40, "memory_mb": 110.0},
            {"timestamp": current_time - 30, "memory_mb": 120.0},
            {"timestamp": current_time - 20, "memory_mb": 130.0},
            {"timestamp": current_time - 10, "memory_mb": 140.0},
        ]
        self.memory_manager._memory_snapshots = snapshots

        forecast = self.memory_manager.get_memory_forecast()

        assert forecast["forecast_available"]
        assert "forecast_memory_mb" in forecast
        assert "trend_mb_per_second" in forecast

    @pytest.mark.timeout(15)
    def test_memory_cleanup_timeout(self):
        """Test memory cleanup completes within timeout."""
        # Set low limit for testing
        self.memory_manager.set_memory_thresholds(hard_limit_mb=50.0)

        # Allocate memory then trigger cleanup
        large_objects = [bytearray(10 * 1024 * 1024) for _ in range(10)]  # 100MB total

        # Cleanup should complete within timeout
        start_time = time.time()
        result = self.memory_manager.force_cleanup(timeout=10.0)
        cleanup_time = time.time() - start_time

        assert result  # Should succeed
        assert cleanup_time < 11.0  # Should respect timeout

    @pytest.mark.timeout(15)
    def test_emergency_cleanup_procedures(self):
        """Test emergency cleanup procedures are executed."""
        # Mock some objects in pools
        self.memory_manager._object_pools["test_pool"] = [
            MagicMock(_in_use=False),
            MagicMock(_in_use=True),
            MagicMock(_in_use=False)
        ]

        # Trigger emergency cleanup
        self.memory_manager.trigger_emergency_cleanup()

        # Should have cleaned up unused objects
        assert len(self.memory_manager._object_pools["test_pool"]) <= 1  # Only in-use object remains

    @pytest.mark.timeout(15)
    def test_hard_limits_disabled(self):
        """Test behavior when hard limits are disabled."""
        self.memory_manager._hard_limits_enabled = False

        # Should allow any allocation when limits disabled
        assert self.memory_manager.check_hard_limits("any_component", 1000.0)

    @pytest.mark.timeout(15)
    def test_forecasting_disabled(self):
        """Test behavior when forecasting is disabled."""
        self.memory_manager._forecasting_enabled = False

        forecast = self.memory_manager.get_memory_forecast()
        assert not forecast["forecast_available"]

    @pytest.mark.timeout(15)
    def test_memory_threshold_validation(self):
        """Test memory threshold validation and setting."""
        # Set valid thresholds
        self.memory_manager.set_memory_thresholds(
            warning_mb=100.0,
            critical_mb=200.0,
            hard_limit_mb=300.0,
            graceful_degradation_mb=150.0,
            emergency_cleanup_mb=250.0
        )

        thresholds = self.memory_manager._memory_thresholds
        assert thresholds["warning_mb"] == 100.0
        assert thresholds["hard_limit_mb"] == 300.0

    @pytest.mark.timeout(15)
    def test_component_limit_defaults(self):
        """Test component limit defaults are properly set."""
        limits = self.memory_manager._component_limits

        assert "cache" in limits
        assert "ml_models" in limits
        assert "default" in limits
        assert limits["default"] == 50.0

    @pytest.mark.timeout(15)
    def test_memory_stats_include_limits(self):
        """Test memory stats include limit information."""
        stats = self.memory_manager.get_memory_stats()

        assert "thresholds" in stats
        assert "hard_limit_mb" in stats["thresholds"]
        assert "graceful_degradation_mb" in stats["thresholds"]

    @pytest.mark.timeout(30)
    def test_large_memory_allocation_simulation(self):
        """Test handling of large memory allocations."""
        # Simulate large allocation
        large_allocation = bytearray(50 * 1024 * 1024)  # 50MB

        # Track it
        self.memory_manager.allocate_memory("large_test", 50.0)

        usage = self.memory_manager.get_component_memory_usage()
        assert usage["large_test"] == 50.0

        # Clean up
        del large_allocation
        gc.collect()

    @pytest.mark.timeout(15)
    def test_concurrent_memory_operations(self):
        """Test memory operations under concurrent access."""
        import threading
        import queue

        results = queue.Queue()

        def allocate_worker(component, amount):
            try:
                result = self.memory_manager.allocate_memory(component, amount)
                results.put(("allocate", component, result))
            except Exception as e:
                results.put(("error", component, str(e)))

        # Start multiple allocation threads
        threads = []
        for i in range(5):
            t = threading.Thread(
                target=allocate_worker,
                args=(f"component_{i}", 5.0)
            )
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=5.0)

        # Check results
        results_list = []
        while not results.empty():
            results_list.append(results.get())

        # Should have 5 successful allocations
        successful = [r for r in results_list if r[0] == "allocate" and r[2]]
        assert len(successful) == 5

    @pytest.mark.timeout(15)
    def test_memory_pressure_detection(self):
        """Test detection of memory pressure conditions."""
        # Set thresholds
        self.memory_manager.set_memory_thresholds(
            warning_mb=100.0,
            graceful_degradation_mb=150.0,
            critical_mb=200.0
        )

        # Test different pressure levels
        test_cases = [
            (80.0, "normal"),
            (120.0, "warning"),
            (160.0, "graceful_degradation"),
            (220.0, "critical")
        ]

        for memory_mb, expected_level in test_cases:
            with patch.object(self.memory_manager, 'get_memory_usage', return_value=memory_mb):
                # Reset state
                self.memory_manager._degradation_active = False
                self.memory_manager._emergency_mode = False

                self.memory_manager._check_memory_usage()

                if expected_level == "graceful_degradation":
                    assert self.memory_manager._degradation_active
                elif expected_level == "critical":
                    assert self.memory_manager._emergency_mode
