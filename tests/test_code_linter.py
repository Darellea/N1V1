"""Unit tests for code_linter module"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import subprocess
from pathlib import Path

# Add auditor to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auditor.code_linter import CodeLinter


class TestCodeLinter(unittest.TestCase):
    """Test cases for CodeLinter class"""

    def setUp(self):
        """Set up test fixtures"""
        self.linter = CodeLinter()

    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """Test successful command execution"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="success output",
            stderr="error output"
        )

        returncode, stdout, stderr = self.linter.run_command(['echo', 'test'])

        self.assertEqual(returncode, 0)
        self.assertEqual(stdout, "success output")
        self.assertEqual(stderr, "error output")
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_run_command_timeout(self, mock_run):
        """Test command timeout handling"""
        mock_run.side_effect = subprocess.TimeoutExpired(['long_command'], 300)

        returncode, stdout, stderr = self.linter.run_command(['long_command'])

        self.assertEqual(returncode, -1)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "Command timed out")

    @patch('subprocess.run')
    def test_run_command_not_found(self, mock_run):
        """Test command not found handling"""
        mock_run.side_effect = FileNotFoundError()

        returncode, stdout, stderr = self.linter.run_command(['nonexistent_command'])

        self.assertEqual(returncode, -1)
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "Command not found: nonexistent_command")

    def test_run_tool_unknown_tool(self):
        """Test running unknown tool"""
        result = self.linter.run_tool('unknown_tool', 'check', '.')

        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['tool'], 'unknown_tool')
        self.assertIn('Unknown tool', result['message'])

    def test_run_tool_no_mode(self):
        """Test running tool with unsupported mode"""
        result = self.linter.run_tool('black', 'unsupported_mode', '.')

        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['tool'], 'black')
        self.assertIn('No unsupported_mode mode', result['message'])

    @patch('auditor.code_linter.CodeLinter.run_command')
    def test_run_tool_black_check_success(self, mock_run_cmd):
        """Test successful black check"""
        mock_run_cmd.return_value = (0, "All done!", "")

        result = self.linter.run_tool('black', 'check', '.')

        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['tool'], 'black')
        self.assertEqual(result['mode'], 'check')
        mock_run_cmd.assert_called_once()

    @patch('auditor.code_linter.CodeLinter.run_command')
    def test_run_tool_black_check_failed(self, mock_run_cmd):
        """Test failed black check"""
        mock_run_cmd.return_value = (1, "would reformat file.py", "")

        result = self.linter.run_tool('black', 'check', '.')

        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['tool'], 'black')
        self.assertEqual(result['returncode'], 1)

    @patch('auditor.code_linter.CodeLinter.run_tool')
    def test_run_linters_all_pass(self, mock_run_tool):
        """Test running all linters when all pass"""
        mock_run_tool.return_value = {
            'status': 'passed',
            'tool': 'black',
            'mode': 'check'
        }

        results = self.linter.run_linters('.', 'check', ['black', 'isort'])

        self.assertEqual(len(results), 3)  # 2 tools + summary
        self.assertEqual(results['summary']['total'], 2)
        self.assertEqual(results['summary']['passed'], 2)
        self.assertEqual(results['summary']['failed'], 0)

    @patch('auditor.code_linter.CodeLinter.run_tool')
    def test_run_linters_some_fail(self, mock_run_tool):
        """Test running all linters when some fail"""
        def mock_tool_response(tool, mode, target):
            if tool == 'black':
                return {'status': 'passed', 'tool': 'black', 'mode': mode}
            elif tool == 'isort':
                return {'status': 'failed', 'tool': 'isort', 'mode': mode, 'stdout': 'import error'}
            else:
                return {'status': 'passed', 'tool': tool, 'mode': mode}

        mock_run_tool.side_effect = mock_tool_response

        results = self.linter.run_linters('.', 'check', ['black', 'isort'])

        self.assertEqual(results['summary']['total'], 2)
        self.assertEqual(results['summary']['passed'], 1)
        self.assertEqual(results['summary']['failed'], 1)

    @patch('auditor.code_linter.CodeLinter.run_tool')
    def test_run_linters_default_tools(self, mock_run_tool):
        """Test running linters with default tools"""
        mock_run_tool.return_value = {'status': 'passed', 'tool': 'black', 'mode': 'check'}

        results = self.linter.run_linters('.', 'check')

        # Should call run_tool for black, isort, ruff (default tools)
        self.assertEqual(mock_run_tool.call_count, 3)

    def test_run_linters_with_target_directory(self):
        """Test running linters on specific target directory"""
        with patch('auditor.code_linter.CodeLinter.run_tool') as mock_run_tool:
            mock_run_tool.return_value = {'status': 'passed', 'tool': 'black', 'mode': 'check'}

            results = self.linter.run_linters('auditor', 'check', ['black'])

            # Check that target was passed correctly
            mock_run_tool.assert_called_with('black', 'check', 'auditor')


class TestCodeLinterIntegration(unittest.TestCase):
    """Integration tests for CodeLinter"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.linter = CodeLinter()

    def test_linter_initialization(self):
        """Test that linter initializes with correct tools"""
        expected_tools = {'black', 'isort', 'ruff', 'flake8'}
        self.assertTrue(expected_tools.issubset(set(self.linter.tools.keys())))

    def test_tool_configurations(self):
        """Test that tool configurations are correct"""
        black_config = self.linter.tools['black']
        self.assertIn('check', black_config)
        self.assertIn('fix', black_config)
        self.assertIsNotNone(black_config['check'])
        self.assertIsNotNone(black_config['fix'])

        # flake8 should have no fix mode
        flake8_config = self.linter.tools['flake8']
        self.assertIn('check', flake8_config)
        self.assertIsNone(flake8_config.get('fix'))


if __name__ == '__main__':
    unittest.main()
