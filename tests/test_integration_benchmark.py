"""
tests/test_integration_benchmark.py

Tests for the updated benchmark helper that no longer swallows exceptions.
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from tests.integration_test_framework import IntegrationTestFramework


class TestBenchmarkExceptionHandling:
    """Test cases for benchmark exception handling."""

    @pytest.fixture
    def framework(self):
        """Create a test framework instance."""
        return IntegrationTestFramework()

    def test_benchmark_successful_operation(self, framework):
        """Test that successful operations are benchmarked correctly."""
        call_count = 0

        async def successful_operation():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)  # Small delay to simulate work
            return "success"

        result = asyncio.run(framework._benchmark_operation(successful_operation))

        assert result["operation"] == "successful_operation"
        assert result["iterations"] == 100
        assert result["success_count"] == 100
        assert result["failure_count"] == 0
        assert result["success_rate"] == 1.0
        assert len(result["errors"]) == 0
        assert "avg_time" in result
        assert "min_time" in result
        assert "max_time" in result
        assert call_count == 100

    def test_benchmark_operation_with_exception_strict_mode(self, framework):
        """Test that exceptions in strict mode are raised immediately."""
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # Fail on first call
                raise ValueError("Test error")
            return "success"

        with pytest.raises(ValueError, match="Test error"):
            asyncio.run(framework._benchmark_operation(failing_operation, strict_mode=True))

        # Should have failed on first iteration
        assert call_count == 1

    def test_benchmark_operation_with_exception_relaxed_mode(self, framework):
        """Test that exceptions in relaxed mode are recorded but don't stop benchmarking."""
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count <= 50:  # Fail first 50 calls
                raise RuntimeError(f"Error on call {call_count}")
            return "success"

        result = asyncio.run(framework._benchmark_operation(failing_operation, strict_mode=False))

        assert result["operation"] == "failing_operation"
        assert result["iterations"] == 100
        assert result["success_count"] == 50  # Last 50 should succeed
        assert result["failure_count"] == 50  # First 50 should fail
        assert result["success_rate"] == 0.5
        assert len(result["errors"]) == 50

        # Check that errors are recorded with correct details
        assert result["errors"][0] == (0, "Error on call 1", "RuntimeError")
        assert result["errors"][49] == (49, "Error on call 50", "RuntimeError")

        # Should have timing metrics for both success and failure
        assert "avg_time" in result  # Successful operations
        assert "avg_failure_time" in result  # Failed operations
        assert call_count == 100

    def test_benchmark_mixed_success_failure(self, framework):
        """Test benchmark with a mix of successful and failed operations."""
        call_count = 0

        async def mixed_operation():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every 3rd call fails
                raise ConnectionError("Network timeout")
            await asyncio.sleep(0.001)
            return f"result_{call_count}"

        result = asyncio.run(framework._benchmark_operation(mixed_operation, strict_mode=False))

        expected_failures = 100 // 3  # 33 failures (iterations 3, 6, 9, ..., 99)
        expected_successes = 100 - expected_failures  # 67 successes

        assert result["success_count"] == expected_successes
        assert result["failure_count"] == expected_failures
        assert result["success_rate"] == expected_successes / 100
        assert len(result["errors"]) == expected_failures

        # Check error details
        for i, (iteration, error_msg, error_type) in enumerate(result["errors"]):
            expected_iteration = (i + 1) * 3  # 3, 6, 9, ...
            assert iteration == expected_iteration - 1  # 0-based indexing
            assert "Network timeout" in error_msg
            assert error_type == "ConnectionError"

        assert call_count == 100

    def test_benchmark_all_failures_relaxed_mode(self, framework):
        """Test benchmark where all operations fail in relaxed mode."""
        async def always_failing_operation():
            raise Exception("Always fails")

        result = asyncio.run(framework._benchmark_operation(always_failing_operation, strict_mode=False))

        assert result["iterations"] == 100
        assert result["success_count"] == 0
        assert result["failure_count"] == 100
        assert result["success_rate"] == 0.0
        assert len(result["errors"]) == 100

        # Should have failure timing metrics but no success metrics
        assert "avg_time" not in result  # No successful operations
        assert "avg_failure_time" in result
        assert "total_failed_time" in result

    def test_benchmark_different_exception_types(self, framework):
        """Test benchmark with different types of exceptions."""
        call_count = 0

        async def varied_exceptions():
            nonlocal call_count
            call_count += 1
            if call_count % 4 == 1:
                raise ValueError("Value error")
            elif call_count % 4 == 2:
                raise RuntimeError("Runtime error")
            elif call_count % 4 == 3:
                raise ConnectionError("Connection error")
            else:
                return "success"

        result = asyncio.run(framework._benchmark_operation(varied_exceptions, strict_mode=False))

        # Should have 25 of each error type (100 / 4 = 25)
        assert result["success_count"] == 25
        assert result["failure_count"] == 75
        assert len(result["errors"]) == 75

        # Check that different error types are recorded
        error_types = [error[2] for error in result["errors"]]
        assert "ValueError" in error_types
        assert "RuntimeError" in error_types
        assert "ConnectionError" in error_types

    def test_benchmark_performance_metrics_structure(self, framework):
        """Test that benchmark results have the correct structure."""
        async def fast_operation():
            return "fast"

        result = asyncio.run(framework._benchmark_operation(fast_operation))

        # Required fields
        required_fields = [
            "operation", "iterations", "success_count", "failure_count",
            "success_rate", "errors", "avg_time", "min_time", "max_time",
            "std_time", "total_successful_time", "overall_avg_time", "overall_total_time"
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Check data types
        assert isinstance(result["operation"], str)
        assert isinstance(result["iterations"], int)
        assert isinstance(result["success_count"], int)
        assert isinstance(result["failure_count"], int)
        assert isinstance(result["success_rate"], float)
        assert isinstance(result["errors"], list)
        assert isinstance(result["avg_time"], (int, float))
        assert isinstance(result["overall_total_time"], (int, float))


class TestBenchmarkIntegration:
    """Test integration of benchmark functionality."""

    @pytest.fixture
    def framework(self):
        """Create a test framework instance."""
        return IntegrationTestFramework()

    def test_run_performance_benchmark_strict_mode(self, framework):
        """Test running performance benchmark in strict mode."""
        async def reliable_operation():
            await asyncio.sleep(0.001)
            return "ok"

        result = asyncio.run(framework.run_performance_benchmark(
            "test_benchmark_strict",
            [reliable_operation],
            strict_mode=True
        ))

        assert result["benchmark"] == "test_benchmark_strict"
        assert result["strict_mode"] is True
        assert len(result["operation_results"]) == 1

        op_result = result["operation_results"][0]
        assert op_result["success_count"] == 100
        assert op_result["failure_count"] == 0

        # Check aggregate metrics
        agg = result["aggregate_metrics"]
        assert agg["total_operations"] == 100
        assert agg["total_successful"] == 100
        assert agg["total_failed"] == 0
        assert agg["overall_success_rate"] == 1.0

    def test_run_performance_benchmark_relaxed_mode(self, framework):
        """Test running performance benchmark in relaxed mode."""
        call_count = 0

        async def unreliable_operation():
            nonlocal call_count
            call_count += 1
            if call_count % 10 == 0:  # Every 10th call fails
                raise ValueError("Occasional failure")
            await asyncio.sleep(0.001)
            return "ok"

        result = asyncio.run(framework.run_performance_benchmark(
            "test_benchmark_relaxed",
            [unreliable_operation],
            strict_mode=False
        ))

        assert result["benchmark"] == "test_benchmark_relaxed"
        assert result["strict_mode"] is False
        assert len(result["operation_results"]) == 1

        op_result = result["operation_results"][0]
        assert op_result["success_count"] == 90  # 90 successful (100 - 10 failures)
        assert op_result["failure_count"] == 10  # 10 failures
        assert len(op_result["errors"]) == 10

        # Check aggregate metrics
        agg = result["aggregate_metrics"]
        assert agg["total_operations"] == 100
        assert agg["total_successful"] == 90
        assert agg["total_failed"] == 10
        assert agg["overall_success_rate"] == 0.9

    def test_run_performance_benchmark_multiple_operations(self, framework):
        """Test running benchmark with multiple operations."""
        async def fast_operation():
            return "fast"

        async def slow_operation():
            await asyncio.sleep(0.01)
            return "slow"

        result = asyncio.run(framework.run_performance_benchmark(
            "test_multiple_ops",
            [fast_operation, slow_operation],
            strict_mode=True
        ))

        assert len(result["operation_results"]) == 2

        # Check that both operations are represented
        op_names = [op["operation"] for op in result["operation_results"]]
        assert "fast_operation" in op_names
        assert "slow_operation" in op_names

        # Check aggregate metrics across both operations
        agg = result["aggregate_metrics"]
        assert agg["total_operations"] == 200  # 100 iterations * 2 operations
        assert agg["total_successful"] == 200
        assert agg["total_failed"] == 0
