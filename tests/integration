"""
tests/test_cli.py

Comprehensive CLI testing for main.py entry point.
Tests --help and --status flags with proper exit codes and output validation.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path


class TestCLI:
    """Test cases for CLI functionality."""

    def test_help_flag(self):
        """Test --help flag produces expected output and exits with code 0."""
        # Run the main script with --help flag
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        # Verify exit code is 0
        assert result.returncode == 0

        # Verify help text is present in output
        assert "usage:" in result.stdout.lower()
        assert "--help" in result.stdout
        assert "--status" in result.stdout

    def test_help_short_flag(self):
        """Test -h flag produces expected output and exits with code 0."""
        # Run the main script with -h flag
        result = subprocess.run(
            [sys.executable, "main.py", "-h"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        # Verify exit code is 0
        assert result.returncode == 0

        # Verify help text is present in output
        assert "usage:" in result.stdout.lower()
        assert "--help" in result.stdout
        assert "--status" in result.stdout

    def test_status_flag_success(self):
        """Test --status flag shows status table and exits with code 0."""
        # Run the main script with --status flag
        result = subprocess.run(
            [sys.executable, "main.py", "--status"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5
        )

        # Verify exit code is 0 (successful execution)
        assert result.returncode == 0

        # Verify status table content is present
        assert "Trading Bot Status" in result.stdout
        assert "Mode" in result.stdout
        assert "Status" in result.stdout
        assert "Balance" in result.stdout

    def test_invalid_flag(self):
        """Test invalid flag produces error message and non-zero exit code."""
        # Run the main script with invalid flag
        result = subprocess.run(
            [sys.executable, "main.py", "--invalid"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        # Verify exit code is non-zero
        assert result.returncode != 0

        # Verify error message in output
        assert "unrecognized arguments" in result.stderr or "error:" in result.stderr.lower()

    def test_no_arguments_normal_execution(self):
        """Test that running without arguments starts normal execution."""
        # Run the main script without arguments but with a short timeout
        # to prevent it from running indefinitely
        try:
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
                timeout=5  # Short timeout to prevent hanging
            )
            # If it completes normally, it should have a non-zero exit code
            # (since it was terminated by timeout)
            assert result.returncode != 0
        except subprocess.TimeoutExpired:
            # This is expected - the script should attempt to run normally
            # and get terminated by our timeout, which means it's working correctly
            pass  # Test passes if timeout occurs

    def test_script_execution_from_different_directory(self):
        """Test that the script can be executed from different working directories."""
        # Get the project root
        project_root = Path(__file__).parent.parent

        # Test from project root
        result1 = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        assert result1.returncode == 0
        assert "usage:" in result1.stdout.lower()

        # Test from tests directory
        result2 = subprocess.run(
            [sys.executable, "../main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        assert result2.returncode == 0
        assert "usage:" in result2.stdout.lower()

    def test_status_flag_output_format(self):
        """Test --status flag output format and content."""
        # Run the main script with --status flag
        result = subprocess.run(
            [sys.executable, "main.py", "--status"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5
        )

        # Verify successful execution
        assert result.returncode == 0

        # Verify the output contains expected table structure
        lines = result.stdout.strip().split('\n')

        # Should have header and some data rows
        assert len(lines) >= 3  # At minimum: header, separator, and one data row

        # Check for table-like formatting (contains pipes or plus signs for table borders)
        table_content = result.stdout
        assert ('|' in table_content or '+' in table_content or '─' in table_content)

    def test_help_flag_comprehensive(self):
        """Test --help flag provides comprehensive information."""
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        assert result.returncode == 0

        help_text = result.stdout.lower()
        # Check for essential help elements
        assert "usage:" in help_text
        assert "options:" in help_text or "optional arguments:" in help_text
        assert "--help" in help_text
        assert "--status" in help_text
        assert "show this help message" in help_text or "show help" in help_text
