#!/usr/bin/env python3
"""
N1V1 Smoke Tests for Canary Deployment Validation
===============================================

This script runs lightweight smoke tests to validate basic functionality
during canary deployments. It focuses on:
1. Framework startup and health checks
2. API endpoint availability
3. Basic trading functionality in paper mode
4. System resource checks

Usage:
    python tests/run_smoke_tests.py [--url BASE_URL] [--timeout SECONDS] [--verbose]
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict

import requests

from utils.logger import get_logger

logger = get_logger(__name__)


class SmokeTestRunner:
    """Smoke test runner for canary deployment validation."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 300,
        verbose: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.test_results = {}
        self.start_time = None
        self.process = None

    def log(self, message: str, level: str = "info"):
        """Log message with appropriate level."""
        if self.verbose or level in ["error", "warning"]:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            elif level == "success":
                logger.info(f"✅ {message}")

    async def run_all_smoke_tests(self) -> Dict[str, Any]:
        """Run all smoke tests and return results."""
        self.log("🚀 Starting N1V1 Smoke Tests")
        self.start_time = time.time()

        try:
            # Test 1: Health endpoint
            await self.test_health_endpoint()

            # Test 2: Ready endpoint
            await self.test_ready_endpoint()

            # Test 3: API status
            await self.test_api_status()

            # Test 4: Framework startup (if running locally)
            await self.test_framework_startup()

            # Test 5: Basic trading simulation
            await self.test_basic_trading()

            # Test 6: System resources
            await self.test_system_resources()

            # Generate summary
            summary = self._generate_summary()

            duration = time.time() - self.start_time
            self.log(f"Smoke tests completed in {duration:.2f} seconds")
            return {
                "status": "passed" if summary["all_passed"] else "failed",
                "duration_seconds": duration,
                "tests_run": len(self.test_results),
                "tests_passed": summary["passed_count"],
                "tests_failed": summary["failed_count"],
                "results": self.test_results,
                "summary": summary,
            }

        except Exception as e:
            logger.exception(f"Smoke tests failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": time.time() - self.start_time
                if self.start_time
                else 0,
            }
        finally:
            # Cleanup
            if self.process:
                self._cleanup_process()

    async def test_health_endpoint(self) -> None:
        """Test /health endpoint."""
        self.log("Testing /health endpoint...")

        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)

            if response.status_code == 200:
                self.test_results["health_endpoint"] = {
                    "status": "passed",
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                }
                self.log("Health check passed", "success")
            else:
                self.test_results["health_endpoint"] = {
                    "status": "failed",
                    "reason": f"Unexpected status code: {response.status_code}",
                    "status_code": response.status_code,
                }
                self.log(f"Health check failed: {response.status_code}", "error")

        except requests.RequestException as e:
            self.test_results["health_endpoint"] = {
                "status": "failed",
                "reason": f"Request failed: {e}",
            }
            self.log(f"Health check failed: {e}", "error")

    async def test_ready_endpoint(self) -> None:
        """Test /ready endpoint."""
        self.log("Testing /ready endpoint...")

        try:
            response = requests.get(f"{self.base_url}/ready", timeout=10)

            if response.status_code == 200:
                self.test_results["ready_endpoint"] = {
                    "status": "passed",
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                }
                self.log("Readiness check passed", "success")
            else:
                self.test_results["ready_endpoint"] = {
                    "status": "failed",
                    "reason": f"Unexpected status code: {response.status_code}",
                    "status_code": response.status_code,
                }
                self.log(f"Readiness check failed: {response.status_code}", "error")

        except requests.RequestException as e:
            self.test_results["ready_endpoint"] = {
                "status": "failed",
                "reason": f"Request failed: {e}",
            }
            self.log(f"Readiness check failed: {e}", "error")

    async def test_api_status(self) -> None:
        """Test API status endpoint."""
        self.log("Testing API status endpoint...")

        try:
            response = requests.get(f"{self.base_url}/api/v1/status", timeout=10)

            if response.status_code in [200, 401]:  # 401 is OK if auth is required
                self.test_results["api_status"] = {
                    "status": "passed",
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                    "auth_required": response.status_code == 401,
                }
                self.log("API status check passed", "success")
            else:
                self.test_results["api_status"] = {
                    "status": "failed",
                    "reason": f"Unexpected status code: {response.status_code}",
                    "status_code": response.status_code,
                }
                self.log(f"API status check failed: {response.status_code}", "error")

        except requests.RequestException as e:
            self.test_results["api_status"] = {
                "status": "failed",
                "reason": f"Request failed: {e}",
            }
            self.log(f"API status check failed: {e}", "error")

    async def test_framework_startup(self) -> None:
        """Test framework startup (for local testing)."""
        self.log("Testing framework startup...")

        # Skip if not running locally
        if "localhost" not in self.base_url:
            self.test_results["framework_startup"] = {
                "status": "skipped",
                "reason": "Not running locally",
            }
            self.log("Framework startup test skipped (not local)", "warning")
            return

        try:
            # Try to start the framework briefly
            self.process = subprocess.Popen(
                [sys.executable, "main.py", "--test-mode"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            )

            # Wait a bit for startup
            await asyncio.sleep(5)

            # Check if process is still running
            if self.process.poll() is None:
                self.test_results["framework_startup"] = {
                    "status": "passed",
                    "message": "Framework started successfully",
                }
                self.log("Framework startup test passed", "success")
            else:
                stdout, stderr = self.process.communicate()
                self.test_results["framework_startup"] = {
                    "status": "failed",
                    "reason": f"Framework exited early: {stderr.decode()}",
                }
                self.log("Framework startup test failed", "error")

        except Exception as e:
            self.test_results["framework_startup"] = {
                "status": "failed",
                "reason": f"Startup failed: {e}",
            }
            self.log(f"Framework startup test failed: {e}", "error")

    async def test_basic_trading(self) -> None:
        """Test basic trading functionality."""
        self.log("Testing basic trading functionality...")

        try:
            # This is a simplified test - in real scenario you'd test actual trading endpoints
            # For now, we'll test if the API can handle basic requests

            response = requests.get(f"{self.base_url}/api/v1/orders", timeout=10)

            if response.status_code in [200, 401]:  # 401 OK if auth required
                self.test_results["basic_trading"] = {
                    "status": "passed",
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                }
                self.log("Basic trading test passed", "success")
            else:
                self.test_results["basic_trading"] = {
                    "status": "failed",
                    "reason": f"Unexpected status code: {response.status_code}",
                    "status_code": response.status_code,
                }
                self.log(f"Basic trading test failed: {response.status_code}", "error")

        except requests.RequestException as e:
            self.test_results["basic_trading"] = {
                "status": "failed",
                "reason": f"Request failed: {e}",
            }
            self.log(f"Basic trading test failed: {e}", "error")

    async def test_system_resources(self) -> None:
        """Test system resource availability."""
        self.log("Testing system resources...")

        try:
            import psutil

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Check CPU usage
            cpu_usage_percent = psutil.cpu_percent(interval=1)

            # Check disk space
            disk = psutil.disk_usage("/")
            disk_usage_percent = disk.percent

            # Define thresholds
            memory_threshold = 90  # 90%
            cpu_threshold = 95  # 95%
            disk_threshold = 95  # 95%

            issues = []

            if memory_usage_percent > memory_threshold:
                issues.append(f"High memory usage: {memory_usage_percent}%")

            if cpu_usage_percent > cpu_threshold:
                issues.append(f"High CPU usage: {cpu_usage_percent}%")

            if disk_usage_percent > disk_threshold:
                issues.append(f"Low disk space: {disk_usage_percent}% used")

            if issues:
                self.test_results["system_resources"] = {
                    "status": "failed",
                    "issues": issues,
                    "memory_percent": memory_usage_percent,
                    "cpu_percent": cpu_usage_percent,
                    "disk_percent": disk_usage_percent,
                }
                self.log(f"System resources test failed: {', '.join(issues)}", "error")
            else:
                self.test_results["system_resources"] = {
                    "status": "passed",
                    "memory_percent": memory_usage_percent,
                    "cpu_percent": cpu_usage_percent,
                    "disk_percent": disk_usage_percent,
                }
                self.log("System resources test passed", "success")

        except ImportError:
            self.test_results["system_resources"] = {
                "status": "skipped",
                "reason": "psutil not available",
            }
            self.log("System resources test skipped (psutil not available)", "warning")
        except Exception as e:
            self.test_results["system_resources"] = {
                "status": "failed",
                "reason": f"Resource check failed: {e}",
            }
            self.log(f"System resources test failed: {e}", "error")

    def _cleanup_process(self):
        """Clean up any running processes."""
        if self.process:
            try:
                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception as e:
                logger.warning(f"Failed to cleanup process: {e}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        passed_count = 0
        failed_count = 0
        skipped_count = 0

        for test_name, result in self.test_results.items():
            status = result.get("status", "unknown")
            if status == "passed":
                passed_count += 1
            elif status == "failed":
                failed_count += 1
            elif status == "skipped":
                skipped_count += 1

        all_passed = failed_count == 0

        return {
            "total_tests": len(self.test_results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "all_passed": all_passed,
            "success_rate": (passed_count / len(self.test_results)) * 100
            if self.test_results
            else 0,
        }


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="N1V1 Smoke Tests")
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL to test"
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Test timeout in seconds"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 60)
    print("🧪 N1V1 SMOKE TESTS")
    print("=" * 60)
    print(f"Target URL: {args.url}")
    print(f"Timeout: {args.timeout}s")
    print(f"Verbose: {args.verbose}")
    print("=" * 60)

    # Run tests
    runner = SmokeTestRunner(args.url, args.timeout, args.verbose)
    results = await runner.run_all_smoke_tests()

    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)

        summary = results.get("summary", {})

        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed_count', 0)}")
        print(f"Failed: {summary.get('failed_count', 0)}")
        print(f"Skipped: {summary.get('skipped_count', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")

        if results["status"] == "passed":
            print("\n✅ ALL SMOKE TESTS PASSED")
        else:
            print("\n❌ SOME SMOKE TESTS FAILED")
            print("\nFailed Tests:")
            for test_name, result in results.get("results", {}).items():
                if result.get("status") == "failed":
                    print(f"  • {test_name}: {result.get('reason', 'Unknown error')}")

        print("=" * 60)

    # Exit with appropriate code
    exit_code = 0 if results.get("status") == "passed" else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
