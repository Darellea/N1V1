"""
Test suite for documentation standardization tools.

Tests the automated docstring validation, standardization, and quality assessment
functionality of the N1V1 Framework.
"""

import pytest
from utils.docstring_standardizer import (
    DocstringStandardizer,
    analyze_documentation,
    standardize_docstring,
    validate_docstring,
    get_docstring_standardizer
)


class TestDocstringStandardizer:
    """Test cases for the DocstringStandardizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.standardizer = DocstringStandardizer()

    def test_initialization(self):
        """Test that the standardizer initializes correctly."""
        assert self.standardizer.issues == []
        assert self.standardizer.metrics.total_functions == 0
        assert self.standardizer.processed_files == set()

    def test_google_format_detection(self):
        """Test detection of Google format docstrings."""
        google_docstring = '''
        """Process trading signals and generate orders.

        This function takes a list of trading signals and converts them
        into executable orders based on the current market conditions.

        Args:
            signals: List of TradingSignal objects to process
            market_data: Current market data for validation
            risk_limits: Risk management constraints

        Returns:
            List of Order objects ready for execution

        Raises:
            ValueError: If signals are invalid
            ConnectionError: If market data is unavailable
        """
        '''

        assert self.standardizer._is_google_format(google_docstring)

    def test_numpy_format_detection(self):
        """Test detection of NumPy format docstrings."""
        numpy_docstring = '''
        """Process trading signals.

        Parameters
        ----------
        signals : list
            List of trading signals
        market_data : dict
            Current market data

        Returns
        -------
        list
            List of orders
        """
        '''

        assert self.standardizer._is_numpy_format(numpy_docstring)

    def test_docstring_completeness_calculation(self):
        """Test calculation of docstring completeness."""
        # Complete docstring
        complete_docstring = '''
        """Function description.

        Args:
            param1: First parameter
            param2: Second parameter

        Returns:
            Result description

        Raises:
            ValueError: When invalid input
        """
        '''

        completeness = self.standardizer._check_docstring_completeness(complete_docstring)
        assert completeness >= 0.8

        # Incomplete docstring
        incomplete_docstring = '''"""Simple description."""'''
        completeness = self.standardizer._check_docstring_completeness(incomplete_docstring)
        assert completeness < 0.5

    def test_docstring_validation(self):
        """Test docstring validation functionality."""
        # Valid docstring
        valid_docstring = '''
        """Process trading data.

        Args:
            data: Input trading data

        Returns:
            Processed data
        """
        '''

        result = self.standardizer.validate_docstring_format(valid_docstring)
        assert result["is_valid"] is True
        assert result["format"] == "google"

        # Invalid docstring (missing)
        result = self.standardizer.validate_docstring_format("")
        assert result["is_valid"] is False
        assert "Missing docstring" in result["issues"]

    def test_docstring_standardization(self):
        """Test docstring standardization."""
        # Original docstring
        original = '''"""Simple function."""'''

        # Standardized version
        standardized = self.standardizer.standardize_docstring(
            original,
            function_name="process_data",
            args=["data", "config"],
            returns="Processed result",
            raises=["ValueError"]
        )

        assert "Args:" in standardized
        assert "Returns:" in standardized
        assert "Raises:" in standardized
        assert "process_data" in standardized

    def test_template_generation(self):
        """Test template docstring generation."""
        template = self.standardizer._generate_template_docstring(
            function_name="calculate_pnl",
            args=["trades", "fees"],
            returns="Total profit and loss",
            raises=["TypeError"]
        )

        assert "calculate_pnl" in template
        assert "Args:" in template
        assert "trades:" in template
        assert "fees:" in template
        assert "Returns:" in template
        assert "Raises:" in template

    def test_format_conversion(self):
        """Test conversion between docstring formats."""
        numpy_docstring = '''
        """Calculate returns.

        Parameters
        ----------
        prices : list
            List of prices
        periods : int
            Number of periods

        Returns
        -------
        float
            Calculated returns
        """
        '''

        google_version = self.standardizer._convert_numpy_to_google(numpy_docstring)
        assert "Args:" in google_version
        assert "Returns:" in google_version


class TestDocstringStandardizerIntegration:
    """Integration tests for the documentation standardizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.standardizer = DocstringStandardizer()

    def test_analyze_codebase_documentation(self):
        """Test full codebase documentation analysis."""
        results = self.standardizer.analyze_codebase_documentation()

        # Check that results contain expected keys
        required_keys = [
            "files_analyzed", "metrics", "issues",
            "recommendations", "quality_score"
        ]

        for key in required_keys:
            assert key in results

        # Check metrics structure
        metrics = results["metrics"]
        assert "total_functions" in metrics
        assert "documentation_coverage" in metrics
        assert isinstance(metrics["documentation_coverage"], str)

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # Create some mock issues
        self.standardizer._create_issue(
            "test.py", 1, "test_func", "missing", "Missing docstring",
            "medium", "Add docstring"
        )

        score = self.standardizer._calculate_quality_score()
        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_recommendations_generation(self):
        """Test generation of improvement recommendations."""
        recommendations = self.standardizer._generate_recommendations()

        assert isinstance(recommendations, list)
        # Should always have some base recommendations
        assert len(recommendations) > 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_get_docstring_standardizer(self):
        """Test singleton pattern for standardizer."""
        std1 = get_docstring_standardizer()
        std2 = get_docstring_standardizer()

        assert std1 is std2
        assert isinstance(std1, DocstringStandardizer)

    def test_analyze_documentation_function(self):
        """Test the analyze_documentation convenience function."""
        results = analyze_documentation()

        assert isinstance(results, dict)
        assert "files_analyzed" in results

    def test_standardize_docstring_function(self):
        """Test the standardize_docstring convenience function."""
        original = '''"""Test function."""'''

        result = standardize_docstring(
            original,
            function_name="test_func",
            args=["param"],
            returns="result"
        )

        assert isinstance(result, str)
        assert len(result) > len(original)

    def test_validate_docstring_function(self):
        """Test the validate_docstring convenience function."""
        docstring = '''
        """Test function.

        Args:
            param: A parameter

        Returns:
            Result
        """
        '''

        result = validate_docstring(docstring)

        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "format" in result


class TestDocstringQualityMetrics:
    """Test documentation quality metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.standardizer = DocstringStandardizer()

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        metrics = self.standardizer.metrics

        assert metrics.total_functions == 0
        assert metrics.documented_functions == 0
        assert metrics.documentation_coverage == 0.0
        assert metrics.average_docstring_length == 0.0

    def test_metrics_calculation(self):
        """Test metrics calculation with sample data."""
        # Simulate some documented functions
        self.standardizer.metrics.total_functions = 10
        self.standardizer.metrics.documented_functions = 7
        self.standardizer.metrics.average_docstring_length = 45.5

        self.standardizer._calculate_final_metrics()

        expected_coverage = 70.0
        assert abs(self.standardizer.metrics.documentation_coverage - expected_coverage) < 0.1

    def test_metrics_to_dict_conversion(self):
        """Test conversion of metrics to dictionary."""
        metrics_dict = self.standardizer._metrics_to_dict()

        required_keys = [
            "total_functions", "documented_functions", "documentation_coverage",
            "google_format_compliance", "numpy_format_compliance"
        ]

        for key in required_keys:
            assert key in metrics_dict


if __name__ == "__main__":
    # Run basic functionality tests
    print("Testing DocstringStandardizer functionality...")

    standardizer = DocstringStandardizer()

    # Test basic validation
    test_docstring = '''
    """Test function.

    Args:
        param1: First parameter
        param2: Second parameter

    Returns:
        Result of the function
    """
    '''

    result = standardizer.validate_docstring_format(test_docstring)
    print(f"Validation result: {result}")

    # Test standardization
    original = '''"""Simple function."""'''
    standardized = standardizer.standardize_docstring(
        original,
        function_name="process_data",
        args=["input_data"],
        returns="Processed data"
    )
    print(f"Standardized docstring:\n{standardized}")

    # Test analysis
    print("Running codebase analysis...")
    results = standardizer.analyze_codebase_documentation()
    print(f"Analysis complete. Files analyzed: {results['files_analyzed']}")
    print(f"Quality score: {results['quality_score']:.1f}/100")

    print("✅ All basic tests passed!")
