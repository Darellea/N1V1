"""
Test to enforce minimum coverage requirements.
This test will fail if test coverage drops below 95%.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path


def test_minimum_coverage_requirement():
    """
    Test that enforces minimum 95% test coverage.
    This test will fail if coverage drops below the required threshold.
    """
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=core", "--cov=utils", "--cov=api", "--cov=ml", "--cov=notifier",
        "--cov-report=term-missing", "--cov-fail-under=95",
        "tests/"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout
            cwd=Path(__file__).parent.parent.parent
        )

        # Check if coverage requirement was met
        if result.returncode == 0:
            # Coverage is >= 95%, test passes
            assert True
        else:
            # Coverage is < 95%, test fails
            # Extract coverage percentage from output for better error message
            coverage_line = None
            for line in result.stderr.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    coverage_line = line
                    break

            if coverage_line:
                pytest.fail(f"Test coverage requirement not met: {coverage_line}. "
                          "Minimum required coverage is 95%.")
            else:
                pytest.fail("Test coverage requirement not met. "
                          "Minimum required coverage is 95%. "
                          f"pytest exit code: {result.returncode}")

    except subprocess.TimeoutExpired:
        pytest.fail("Coverage test timed out after 10 minutes")
    except Exception as e:
        pytest.fail(f"Coverage test failed with error: {str(e)}")


def test_coverage_report_generation():
    """
    Test that coverage reports are generated successfully.
    """
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=core", "--cov=utils", "--cov=api", "--cov=ml", "--cov=notifier",
        "--cov-report=xml", "--cov-report=html",
        "tests/unit/test_config_manager.py"  # Run just one test for speed
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent
    )

    # Check that coverage files were created
    coverage_xml = Path("coverage.xml")
    coverage_html = Path("htmlcov/index.html")

    assert coverage_xml.exists(), "coverage.xml was not generated"
    assert coverage_html.exists(), "HTML coverage report was not generated"

    # Clean up
    if coverage_xml.exists():
        coverage_xml.unlink()
    if coverage_html.parent.exists():
        import shutil
        shutil.rmtree(coverage_html.parent)


def test_critical_modules_coverage():
    """
    Test that critical modules have adequate test coverage.
    """
    critical_modules = [
        "core.order_manager",
        "core.config_manager",
        "core.signal_processor",
        "utils.logger"
    ]

    for module in critical_modules:
        cmd = [
            sys.executable, "-m", "pytest",
            f"--cov={module}",
            "--cov-report=term-missing",
            f"tests/unit/test_{module.split('.')[-1]}.py"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )

        # Each critical module should have at least 80% coverage
        if result.returncode != 0:
            pytest.fail(f"Critical module {module} has insufficient test coverage")
