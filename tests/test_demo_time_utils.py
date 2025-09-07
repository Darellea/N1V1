import pytest
import logging
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

# Import the demo function
from demo.demo_time_utils import demo_time_utilities


class TestDemoTimeUtils:
    """Test cases for demo_time_utils.py functionality."""

    @pytest.fixture
    def mock_logger(self):
        """Mock logger to capture log output."""
        with patch('demo.demo_time_utils.logger') as mock_logger:
            yield mock_logger

    @pytest.fixture
    def mock_time_functions(self):
        """Mock time utility functions."""
        with patch('demo.demo_time_utils.now_ms') as mock_now_ms, \
             patch('demo.demo_time_utils.to_ms') as mock_to_ms, \
             patch('demo.demo_time_utils.to_iso') as mock_to_iso:

            # Set up mock return values
            mock_now_ms.return_value = 1672574400000  # 2023-01-01 12:00:00 UTC
            mock_to_iso.return_value = "2023-01-01T12:00:00.000Z"

            def mock_to_ms_func(x):
                if x == 1672574400:
                    return 1672574400000
                elif x == 1672574400000:
                    return 1672574400000
                elif x == 1672574400.123:
                    return 1672574400123
                elif x == "2023-01-01T12:00:00Z":
                    return 1672574400000
                elif x == "1672574400":
                    return 1672574400000
                elif x == "1672574400000":
                    return 1672574400000
                elif x is None:
                    return None
                elif x == "invalid":
                    return None
                elif isinstance(x, (list, dict, set, tuple)):
                    return None
                else:
                    return 1672574400000

            mock_to_ms.side_effect = mock_to_ms_func

            yield {
                'now_ms': mock_now_ms,
                'to_ms': mock_to_ms,
                'to_iso': mock_to_iso
            }

    def test_demo_time_utilities_basic_execution(self, mock_logger, mock_time_functions):
        """Test that demo_time_utilities runs without errors."""
        # Execute the demo function
        demo_time_utilities()

        # Verify that logger.info was called multiple times
        assert mock_logger.info.call_count > 5  # Should log multiple messages

        # Check that key log messages were called
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Verify main sections are logged
        assert any("=== Time Utilities Demo ===" in msg for msg in log_calls)
        assert any("=== Conversion Examples ===" in msg for msg in log_calls)
        assert any("=== Edge Cases ===" in msg for msg in log_calls)
        assert any("=== ISO Conversion ===" in msg for msg in log_calls)

    def test_demo_time_utilities_current_time_logging(self, mock_logger, mock_time_functions):
        """Test logging of current time values."""
        demo_time_utilities()

        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Verify current time logging
        assert any("Current time (ms):" in msg for msg in log_calls)
        assert any("ISO format:" in msg for msg in log_calls)
        assert any("Round-trip ms:" in msg for msg in log_calls)
        assert any("Round-trip difference:" in msg for msg in log_calls)

    def test_demo_time_utilities_conversion_examples(self, mock_logger, mock_time_functions):
        """Test conversion examples section."""
        demo_time_utilities()

        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Verify conversion examples are logged
        conversion_examples = [
            "Seconds as int",
            "Milliseconds as int",
            "Seconds as float",
            "ISO string",
            "Numeric string seconds",
            "Numeric string milliseconds"
        ]

        for example in conversion_examples:
            assert any(example in msg for msg in log_calls), f"Missing conversion example: {example}"

    def test_demo_time_utilities_edge_cases(self, mock_logger, mock_time_functions):
        """Test edge cases section."""
        demo_time_utilities()

        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Verify edge cases are logged
        edge_cases = [
            "None",
            "Invalid string",
            "List (unsupported)"
        ]

        for case in edge_cases:
            assert any(case in msg for msg in log_calls), f"Missing edge case: {case}"

    def test_demo_time_utilities_iso_conversion(self, mock_logger, mock_time_functions):
        """Test ISO conversion examples."""
        demo_time_utilities()

        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Verify ISO conversion examples
        iso_examples = [
            "Epoch",
            "Recent timestamp",
            "Future timestamp"
        ]

        for example in iso_examples:
            assert any(example in msg for msg in log_calls), f"Missing ISO example: {example}"

    def test_demo_time_utilities_function_calls(self, mock_time_functions):
        """Test that the correct utility functions are called."""
        demo_time_utilities()

        # Verify function call counts
        mock_time_functions['now_ms'].assert_called()
        mock_time_functions['to_iso'].assert_called()
        mock_time_functions['to_ms'].assert_called()

        # to_ms should be called multiple times for examples
        assert mock_time_functions['to_ms'].call_count > 5

    def test_demo_time_utilities_with_real_functions(self):
        """Test demo_time_utilities with real utility functions (integration test)."""
        # Capture log output
        log_capture_string = StringIO()
        ch = logging.StreamHandler(log_capture_string)
        ch.setLevel(logging.INFO)

        logger = logging.getLogger('demo.demo_time_utils')
        logger.setLevel(logging.INFO)
        logger.addHandler(ch)

        try:
            # Execute demo with real functions
            demo_time_utilities()

            # Get log output
            log_contents = log_capture_string.getvalue()

            # Verify key sections are present
            assert "=== Time Utilities Demo ===" in log_contents
            assert "Current time (ms):" in log_contents
            assert "ISO format:" in log_contents
            assert "=== Conversion Examples ===" in log_contents
            assert "=== Edge Cases ===" in log_contents
            assert "=== ISO Conversion ===" in log_contents

        finally:
            logger.removeHandler(ch)

    def test_demo_time_utilities_main_execution(self):
        """Test execution when run as main script."""
        # Mock sys.argv and other dependencies
        with patch('sys.argv', ['demo_time_utils.py']), \
             patch('demo.demo_time_utils.demo_time_utilities') as mock_demo:

            # Import and run as main
            import demo.demo_time_utils
            # Simulate the if __name__ == "__main__" block
            demo.demo_time_utils.main()

            # Verify demo function was called
            mock_demo.assert_called_once()

    def test_demo_time_utilities_error_handling(self, mock_logger):
        """Test error handling in demo function."""
        # Mock time functions to raise exceptions
        with patch('demo.demo_time_utils.now_ms', side_effect=Exception("Test error")), \
             patch('demo.demo_time_utils.to_ms'), \
             patch('demo.demo_time_utils.to_iso'):

            # Should not crash, but may log errors
            demo_time_utilities()

            # Verify some logging still occurred
            assert mock_logger.info.call_count >= 1

    def test_demo_time_utilities_comprehensive_logging(self, mock_logger, mock_time_functions):
        """Test comprehensive logging output."""
        demo_time_utilities()

        # Get all log messages
        log_messages = [call[0][0] for call in mock_logger.info.call_args_list]

        # Count different types of messages
        section_headers = sum(1 for msg in log_messages if msg.startswith("===") and msg.endswith("==="))
        time_logs = sum(1 for msg in log_messages if "time" in msg.lower())
        conversion_logs = sum(1 for msg in log_messages if "->" in msg)
        example_logs = sum(1 for msg in log_messages if ":" in msg and "->" in msg)

        # Verify we have expected number of different log types
        assert section_headers >= 3  # At least 3 sections
        assert time_logs >= 3  # Current time, ISO, round-trip
        assert conversion_logs >= 6  # Various conversion examples
        assert example_logs >= 9  # All examples combined

    @pytest.mark.parametrize("input_val,expected_desc", [
        (1672574400, "Seconds as int"),
        (1672574400000, "Milliseconds as int"),
        (1672574400.123, "Seconds as float"),
        ("2023-01-01T12:00:00Z", "ISO string"),
        ("1672574400", "Numeric string seconds"),
        ("1672574400000", "Numeric string milliseconds"),
    ])
    def test_demo_time_utilities_parametrized_examples(self, mock_logger, input_val, expected_desc):
        """Test specific conversion examples with parametrization."""
        with patch('demo.demo_time_utils.now_ms', return_value=1672574400000), \
             patch('demo.demo_time_utils.to_iso', return_value="2023-01-01T12:00:00.000Z"), \
             patch('demo.demo_time_utils.to_ms') as mock_to_ms:

            mock_to_ms.return_value = 1672574400000

            demo_time_utilities()

            # Verify the specific example was logged
            log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any(expected_desc in msg and str(input_val) in msg for msg in log_calls)

    def test_demo_time_utilities_edge_case_logging(self, mock_logger):
        """Test edge case logging specifically."""
        with patch('demo.demo_time_utils.now_ms', return_value=1672574400000), \
             patch('demo.demo_time_utils.to_iso', return_value="2023-01-01T12:00:00.000Z"), \
             patch('demo.demo_time_utils.to_ms') as mock_to_ms:

            # Mock to_ms for edge cases
            mock_to_ms.side_effect = lambda x: None if x in [None, "invalid", [1, 2, 3]] else 1672574400000

            demo_time_utilities()

            log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

            # Verify edge cases are logged
            edge_cases_found = sum(1 for msg in log_calls if any(case in msg for case in ["None", "Invalid string", "List (unsupported)"]))
            assert edge_cases_found >= 3

    def test_demo_time_utilities_iso_examples(self, mock_logger):
        """Test ISO conversion examples specifically."""
        with patch('demo.demo_time_utils.now_ms', return_value=1672574400000), \
             patch('demo.demo_time_utils.to_ms', return_value=1672574400000), \
             patch('demo.demo_time_utils.to_iso', return_value="2023-01-01T12:00:00.000Z") as mock_to_iso:

            demo_time_utilities()

            log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

            # Verify ISO examples
            iso_examples_found = sum(1 for msg in log_calls if any(case in msg for case in ["Epoch", "Recent timestamp", "Future timestamp"]))
            assert iso_examples_found >= 3
