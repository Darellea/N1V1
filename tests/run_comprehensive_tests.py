"""
Comprehensive Test Runner for N1V1
===================================

This script runs categorized tests for N1V1 with support for unit, integration,
stress tests, and smoke tests. It enforces coverage requirements and provides
detailed reporting for CI/CD pipelines.
"""

import argparse
import subprocess
import sys
import time
from typing import Any, Dict, List


class ComprehensiveTestRunner:
    """Comprehensive test runner with categorization support."""

    def __init__(self):
        self.test_results = {}
        self.coverage_data = {}
        self.start_time = None
        self.end_time = None

    def run_tests(self, test_type: str = "all", smoke: bool = False) -> Dict[str, Any]:
        """Run tests based on type or smoke mode."""
        self.start_time = time.time()

        try:
            if smoke:
                return self._run_smoke_tests()
            elif test_type == "unit":
                return self._run_unit_tests()
            elif test_type == "integration":
                return self._run_integration_tests()
            elif test_type == "stress":
                return self._run_stress_tests()
            else:
                return self._run_all_tests()
        except Exception as e:
            return {"error": str(e), "status": "failed"}

    def _run_smoke_tests(self) -> Dict[str, Any]:
        """Run minimal smoke tests for canary deploys."""
        print("🚀 Running Smoke Tests")

        # Define smoke test files/patterns
        smoke_tests = [
            "tests/unit/test_config_manager.py::TestConfigManager::test_basic_config",
            "tests/unit/test_order_manager.py::TestOrderManager::test_order_creation",
            "tests/integration/test_distributed_system.py::TestDistributedSystem::test_basic_connectivity",
            "tests/unit/test_healthcheck.py",  # Assuming health check tests exist
        ]

        # Run smoke tests
        result = self._run_pytest(smoke_tests, timeout=120)

        # Generate coverage for smoke
        coverage_result = self._run_coverage(smoke_tests)

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        summary = self._generate_summary(result, coverage_result, duration)

        return {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": duration,
            "summary": summary,
            "coverage": coverage_result,
        }

    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        print("🔧 Running Unit Tests")
        result = self._run_pytest(["tests/unit/"], timeout=600)
        coverage_result = self._run_coverage(["tests/unit/"])
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        summary = self._generate_summary(result, coverage_result, duration)
        return {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": duration,
            "summary": summary,
            "coverage": coverage_result,
        }

    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("🔗 Running Integration Tests")
        result = self._run_pytest(["tests/integration/"], timeout=900)
        coverage_result = self._run_coverage(["tests/integration/"])
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        summary = self._generate_summary(result, coverage_result, duration)
        return {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": duration,
            "summary": summary,
            "coverage": coverage_result,
        }

    def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests."""
        print("⚡ Running Stress Tests")
        result = self._run_pytest(["tests/stress/"], timeout=1800)
        coverage_result = self._run_coverage(["tests/stress/"])
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        summary = self._generate_summary(result, coverage_result, duration)
        return {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": duration,
            "summary": summary,
            "coverage": coverage_result,
        }

    def _run_all_tests(self) -> Dict[str, Any]:
        """Run all tests."""
        print("🚀 Running All Tests")
        result = self._run_pytest(["tests/"], timeout=3600)
        coverage_result = self._run_coverage(["tests/"])
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        summary = self._generate_summary(result, coverage_result, duration)
        return {
            "status": "passed" if result["returncode"] == 0 else "failed",
            "duration": duration,
            "summary": summary,
            "coverage": coverage_result,
        }

    def _run_pytest(self, test_paths: List[str], timeout: int = 600) -> Dict[str, Any]:
        """Run pytest with given paths."""
        cmd = (
            [sys.executable, "-m", "pytest"]
            + test_paths
            + [
                "-v",
                "--tb=short",
                "--json-report",
                "--json-report-file=test_results.json",
            ]
        )

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"returncode": 1, "stdout": "", "stderr": "Tests timed out"}

    def _run_coverage(self, test_paths: List[str]) -> Dict[str, Any]:
        """Run coverage analysis."""
        cmd = (
            [sys.executable, "-m", "pytest"]
            + test_paths
            + [
                "--cov=core",
                "--cov=utils",
                "--cov=api",
                "--cov=ml",
                "--cov=notifier",
                "--cov-report=xml",
                "--cov-report=term-missing",
                "--cov-fail-under=95",
            ]
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
            coverage_percent = self._extract_coverage(result.stdout)
            return {
                "returncode": result.returncode,
                "coverage": coverage_percent,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": 1,
                "coverage": 0,
                "stdout": "",
                "stderr": "Coverage timed out",
            }

    def _extract_coverage(self, output: str) -> float:
        """Extract coverage percentage from pytest output."""
        for line in output.split("\n"):
            if "TOTAL" in line and "%" in line:
                try:
                    return float(line.split()[-1].rstrip("%"))
                except:
                    pass
        return 0.0

    def _generate_summary(
        self, result: Dict, coverage: Dict, duration: float
    ) -> Dict[str, Any]:
        """Generate test summary."""
        passed = result["returncode"] == 0
        coverage_pct = coverage.get("coverage", 0)

        summary = {
            "passed": passed,
            "failed": not passed,
            "coverage_percent": coverage_pct,
            "duration_seconds": duration,
            "coverage_compliant": coverage_pct >= 95.0,
        }

        if not passed:
            summary["errors"] = result.get("stderr", "")

        return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="N1V1 Comprehensive Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument("--stress", action="store_true", help="Run only stress tests")
    parser.add_argument(
        "--smoke", action="store_true", help="Run smoke tests for canary"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Determine test type
    if args.smoke:
        test_type = "smoke"
    elif args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.stress:
        test_type = "stress"
    else:
        test_type = "all"

    print("=" * 80)
    print("🧪 N1V1 COMPREHENSIVE TEST RUNNER")
    print("=" * 80)
    print(f"Test Type: {test_type.upper()}")
    print("=" * 80)

    runner = ComprehensiveTestRunner()
    results = runner.run_tests(test_type=test_type, smoke=args.smoke)

    # Print results
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS")
    print("=" * 80)

    summary = results.get("summary", {})

    print(f"Status: {'✅ PASSED' if results['status'] == 'passed' else '❌ FAILED'}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")
    print(f"Coverage: {summary.get('coverage_percent', 0):.1f}%")
    print(
        f"Coverage Compliant: {'✅ YES' if summary.get('coverage_compliant', False) else '❌ NO'}"
    )

    if summary.get("failed"):
        print(f"\nErrors:\n{summary.get('errors', '')}")

    # Exit with code
    exit_code = (
        0
        if results["status"] == "passed" and summary.get("coverage_compliant", False)
        else 1
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
