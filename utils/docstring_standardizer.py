"""
Documentation Standardization Tool for N1V1 Framework.

Provides automated docstring validation, standardization to Google/NumPy format,
and comprehensive documentation quality metrics.
"""

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from utils.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


@dataclass
class DocstringIssue:
    """Represents a documentation issue found during analysis."""

    file_path: str
    line_number: int
    function_name: str
    issue_type: str  # missing, incomplete, malformed, inconsistent
    description: str
    severity: str  # critical, high, medium, low
    fix_suggestion: str


@dataclass
class DocstringMetrics:
    """Metrics for documentation quality assessment."""

    total_functions: int = 0
    documented_functions: int = 0
    undocumented_functions: int = 0
    google_format_compliance: int = 0
    numpy_format_compliance: int = 0
    malformed_docstrings: int = 0
    incomplete_docstrings: int = 0
    average_docstring_length: float = 0.0
    documentation_coverage: float = 0.0


class DocstringStandardizer:
    """
    Automated tool for standardizing and validating docstrings across the codebase.

    Supports conversion between different docstring formats and provides
    comprehensive quality metrics and improvement suggestions.
    """

    def __init__(self):
        """
        Initialize the DocstringStandardizer.

        Sets up internal state for tracking documentation issues, metrics,
        and processed files. Configures regex patterns for detecting different
        docstring formats and defines required sections for comprehensive documentation.
        """
        self.issues: List[DocstringIssue] = []
        self.metrics = DocstringMetrics()
        self.processed_files: Set[str] = set()

        # Patterns for different docstring formats
        self.google_pattern = re.compile(r'^\s*"""')
        self.numpy_pattern = re.compile(
            r'^\s*""".*\n\s*Parameters?\n\s*-+\n', re.MULTILINE
        )

        # Required sections for comprehensive docstrings
        self.required_sections = {
            "description",
            "args",
            "returns",
            "raises",
            "examples",
            "notes",
        }

    def analyze_codebase_documentation(self) -> Dict[str, Any]:
        """
        Perform comprehensive documentation analysis of the codebase.

        Returns:
            Dictionary containing analysis results and recommendations
        """
        logger.info("Starting comprehensive documentation analysis")

        python_files = self._get_python_files()

        for file_path in python_files:
            try:
                self._analyze_file_documentation(file_path)
                self.processed_files.add(str(file_path))
            except Exception as e:
                logger.error(f"Error analyzing documentation in {file_path}: {e}")

        # Calculate final metrics
        self._calculate_final_metrics()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        return {
            "files_analyzed": len(self.processed_files),
            "metrics": self._metrics_to_dict(),
            "issues": [self._issue_to_dict(issue) for issue in self.issues],
            "recommendations": recommendations,
            "quality_score": self._calculate_quality_score(),
        }

    def _analyze_file_documentation(self, file_path: Path):
        """
        Analyze documentation in a single Python file.

        Parses the file's AST to identify functions and classes, then analyzes
        their docstrings for quality, completeness, and format compliance.
        Updates internal metrics and issues lists based on findings.

        Args:
            file_path: Path to the Python file to analyze
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    self._analyze_node_documentation(node, file_path)

        except SyntaxError:
            logger.warning(
                f"Syntax error in {file_path}, skipping documentation analysis"
            )
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

    def _analyze_node_documentation(self, node: ast.AST, file_path: Path):
        """Analyze documentation for a specific AST node."""
        self.metrics.total_functions += 1

        # Get function/class name and docstring
        name = getattr(node, "name", "unknown")
        docstring = ast.get_docstring(node)

        if docstring:
            self.metrics.documented_functions += 1
            self._analyze_docstring_quality(
                docstring, name, file_path, getattr(node, "lineno", 0)
            )
        else:
            self.metrics.undocumented_functions += 1
            self._create_issue(
                file_path=str(file_path),
                line_number=getattr(node, "lineno", 0),
                function_name=name,
                issue_type="missing",
                description=f"Missing docstring for {type(node).__name__.lower()} '{name}'",
                severity="medium",
                fix_suggestion="Add comprehensive docstring following Google format",
            )

    def _analyze_docstring_quality(
        self, docstring: str, name: str, file_path: Path, line_number: int
    ):
        """Analyze the quality of a docstring."""
        # Check format compliance
        is_google = self._is_google_format(docstring)
        is_numpy = self._is_numpy_format(docstring)

        if is_google:
            self.metrics.google_format_compliance += 1
        elif is_numpy:
            self.metrics.numpy_format_compliance += 1
        else:
            self.metrics.malformed_docstrings += 1
            self._create_issue(
                file_path=file_path,
                line_number=line_number,
                function_name=name,
                issue_type="malformed",
                description="Docstring format not recognized (Google or NumPy)",
                severity="low",
                fix_suggestion="Convert to Google format with proper sections",
            )

        # Check completeness
        completeness_score = self._check_docstring_completeness(docstring)
        if completeness_score < 0.5:
            self.metrics.incomplete_docstrings += 1
            self._create_issue(
                file_path=file_path,
                line_number=line_number,
                function_name=name,
                issue_type="incomplete",
                description=f"Incomplete docstring (completeness: {completeness_score:.1f})",
                severity="low",
                fix_suggestion="Add missing sections: Args, Returns, Raises",
            )

        # Update average length
        docstring_length = len(docstring.strip())
        if self.metrics.average_docstring_length == 0:
            self.metrics.average_docstring_length = docstring_length
        else:
            # Running average
            total_docs = self.metrics.documented_functions
            self.metrics.average_docstring_length = (
                (self.metrics.average_docstring_length * (total_docs - 1))
                + docstring_length
            ) / total_docs

    def _is_google_format(self, docstring: str) -> bool:
        """Check if docstring follows Google format."""
        lines = docstring.strip().split("\n")

        # Google format typically has Args:, Returns:, etc. on separate lines
        has_args = any("Args:" in line or "Arguments:" in line for line in lines)
        has_returns = any("Returns:" in line for line in lines)

        return has_args or has_returns or len(lines) > 3

    def _is_numpy_format(self, docstring: str) -> bool:
        """Check if docstring follows NumPy format."""
        return "Parameters" in docstring and "----------" in docstring

    def _check_docstring_completeness(self, docstring: str) -> float:
        """Check how complete a docstring is (0.0 to 1.0)."""
        score = 0.0
        doc_lower = docstring.lower()

        # Check for key sections
        if (
            "args:" in doc_lower
            or "arguments:" in doc_lower
            or "parameters" in doc_lower
        ):
            score += 0.3
        if "returns:" in doc_lower:
            score += 0.3
        if "raises:" in doc_lower or "exceptions:" in doc_lower:
            score += 0.2
        if "examples:" in doc_lower or "example" in doc_lower:
            score += 0.1
        if "notes:" in doc_lower or len(docstring.strip()) > 100:
            score += 0.1

        return min(score, 1.0)

    def _create_issue(
        self,
        file_path: str,
        line_number: int,
        function_name: str,
        issue_type: str,
        description: str,
        severity: str,
        fix_suggestion: str,
    ):
        """Create a documentation issue."""
        issue = DocstringIssue(
            file_path=file_path,
            line_number=line_number,
            function_name=function_name,
            issue_type=issue_type,
            description=description,
            severity=severity,
            fix_suggestion=fix_suggestion,
        )
        self.issues.append(issue)

    def _calculate_final_metrics(self):
        """Calculate final documentation metrics."""
        if self.metrics.total_functions > 0:
            self.metrics.documentation_coverage = (
                self.metrics.documented_functions / self.metrics.total_functions
            ) * 100

    def _metrics_to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_functions": self.metrics.total_functions,
            "documented_functions": self.metrics.documented_functions,
            "undocumented_functions": self.metrics.undocumented_functions,
            "documentation_coverage": ".1f",
            "google_format_compliance": self.metrics.google_format_compliance,
            "numpy_format_compliance": self.metrics.numpy_format_compliance,
            "malformed_docstrings": self.metrics.malformed_docstrings,
            "incomplete_docstrings": self.metrics.incomplete_docstrings,
            "average_docstring_length": ".1f",
        }

    def _issue_to_dict(self, issue: DocstringIssue) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            "file_path": issue.file_path,
            "line_number": issue.line_number,
            "function_name": issue.function_name,
            "issue_type": issue.issue_type,
            "description": issue.description,
            "severity": issue.severity,
            "fix_suggestion": issue.fix_suggestion,
        }

    def _calculate_quality_score(self) -> float:
        """Calculate overall documentation quality score (0-100)."""
        if self.metrics.total_functions == 0:
            return 0.0

        base_score = 100.0

        # Coverage penalty
        coverage_penalty = (1 - self.metrics.documentation_coverage / 100) * 30
        base_score -= coverage_penalty

        # Format compliance bonus
        format_compliance = (
            self.metrics.google_format_compliance + self.metrics.numpy_format_compliance
        ) / self.metrics.documented_functions
        format_bonus = format_compliance * 10
        base_score += format_bonus

        # Malformed penalty
        malformed_penalty = self.metrics.malformed_docstrings * 2
        base_score -= malformed_penalty

        # Incomplete penalty
        incomplete_penalty = self.metrics.incomplete_docstrings * 1
        base_score -= incomplete_penalty

        return max(0, min(100, base_score))

    def _generate_recommendations(self) -> List[str]:
        """Generate documentation improvement recommendations."""
        recommendations = []

        coverage = self.metrics.documentation_coverage
        if coverage < 70:
            recommendations.append(
                "🚨 CRITICAL: Documentation coverage below 70% - prioritize adding missing docstrings"
            )
        elif coverage < 85:
            recommendations.append("⚠️ HIGH: Improve documentation coverage to 85%+")

        if self.metrics.malformed_docstrings > 10:
            recommendations.append(
                "🔧 MEDIUM: Standardize docstring formats to Google style"
            )

        if self.metrics.incomplete_docstrings > 20:
            recommendations.append(
                "📝 MEDIUM: Complete missing docstring sections (Args, Returns, Raises)"
            )

        if self.metrics.average_docstring_length < 50:
            recommendations.append("📖 LOW: Improve docstring detail and examples")

        recommendations.extend(
            [
                "🔄 Implement automated docstring validation in CI/CD",
                "📚 Create documentation templates for common patterns",
                "🎯 Set up documentation quality gates for code reviews",
                "📊 Add documentation metrics to development dashboard",
            ]
        )

        return recommendations

    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []
        for pattern in ["*.py"]:
            python_files.extend(PROJECT_ROOT.rglob(pattern))

        # Exclude common directories
        exclude_patterns = [
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            "htmlcov",
            "build",
            "dist",
            "*.egg-info",
            "venv",
            ".venv",
        ]

        filtered_files = []
        for file_path in python_files:
            if not any(excl in str(file_path) for excl in exclude_patterns):
                filtered_files.append(file_path)

        return filtered_files

    def standardize_docstring(
        self,
        docstring: str,
        function_name: str = "",
        args: List[str] = None,
        returns: str = "",
        raises: List[str] = None,
    ) -> str:
        """
        Convert a docstring to standardized Google format.

        Args:
            docstring: Original docstring
            function_name: Name of the function
            args: List of argument names
            returns: Return type description
            raises: List of exceptions that can be raised

        Returns:
            Standardized docstring in Google format
        """
        if not docstring:
            return self._generate_template_docstring(
                function_name, args, returns, raises
            )

        # If already in good format, return as-is
        if self._is_google_format(docstring):
            return docstring

        # Convert from NumPy format if needed
        if self._is_numpy_format(docstring):
            return self._convert_numpy_to_google(docstring)

        # Otherwise, enhance existing docstring
        return self._enhance_docstring(docstring, function_name, args, returns, raises)

    def _generate_template_docstring(
        self, function_name: str, args: List[str], returns: str, raises: List[str]
    ) -> str:
        """Generate a template docstring in Google format."""
        template = f'"""{function_name.capitalize()}'

        if function_name:
            template += f" {function_name}"

        template += ".\n\n"

        if args:
            template += "Args:\n"
            for arg in args:
                template += f"    {arg}: Description of {arg}\n"
            template += "\n"

        if returns:
            template += f"Returns:\n    {returns}\n\n"

        if raises:
            template += "Raises:\n"
            for exc in raises:
                template += f"    {exc}: Description of when {exc} is raised\n"
            template += "\n"

        template += '"""'
        return template

    def _convert_numpy_to_google(self, docstring: str) -> str:
        """Convert NumPy format docstring to Google format."""
        # This is a simplified conversion - full conversion would be more complex
        lines = docstring.split("\n")
        google_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Convert Parameters section
            if line.lower() == "parameters":
                google_lines.append("Args:")
                i += 2  # Skip the dashes
                continue

            # Convert Returns section
            if "returns" in line.lower():
                google_lines.append("Returns:")
                i += 2  # Skip the dashes
                continue

            # Convert Raises section
            if "raises" in line.lower():
                google_lines.append("Raises:")
                i += 2  # Skip the dashes
                continue

            google_lines.append(line)
            i += 1

        return "\n".join(google_lines)

    def _enhance_docstring(
        self,
        docstring: str,
        function_name: str,
        args: List[str],
        returns: str,
        raises: List[str],
    ) -> str:
        """
        Enhance existing docstring with missing sections and standardize format.

        This method creates a standardized Google format docstring that always
        includes the function name in the header, then adds any missing sections
        (Args, Returns, Raises) while preserving existing content.

        Formatting rules enforced:
        - Function name always appears in header as "{function_name} function."
        - Args section lists all parameters with descriptions
        - Returns section describes return value
        - Raises section lists exceptions with conditions

        Args:
            docstring: Original docstring content
            function_name: Name of the function being documented
            args: List of argument names
            returns: Description of return value
            raises: List of exception types that can be raised

        Returns:
            Enhanced docstring in standardized Google format
        """
        # Extract content from existing docstring (remove triple quotes)
        content = docstring.strip()
        if content.startswith('"""') and content.endswith('"""'):
            content = content[3:-3]
        elif content.startswith("'''") and content.endswith("'''"):
            content = content[3:-3]

        lines = content.strip().split("\n")
        enhanced_lines = []

        # Always start with function name in header
        if function_name:
            enhanced_lines.append(f"{function_name} function.")
            enhanced_lines.append("")

        # Add original content if it has meaningful description
        if lines and len(lines[0].strip()) >= 10:
            enhanced_lines.extend(lines)
        elif lines and len(lines) > 1:
            # Keep additional content if present
            enhanced_lines.extend(lines[1:])

        # Add Args section if missing and we have args info
        if args and not any(
            "Args:" in line or "Arguments:" in line for line in enhanced_lines
        ):
            enhanced_lines.append("")
            enhanced_lines.append("Args:")
            for arg in args:
                enhanced_lines.append(f"    {arg}: Description of {arg}")
            enhanced_lines.append("")

        # Add Returns section if missing and we have return info
        if returns and not any("Returns:" in line for line in enhanced_lines):
            enhanced_lines.append("")
            enhanced_lines.append("Returns:")
            enhanced_lines.append(f"    {returns}")
            enhanced_lines.append("")

        # Add Raises section if missing and we have exception info
        if raises and not any("Raises:" in line for line in enhanced_lines):
            enhanced_lines.append("")
            enhanced_lines.append("Raises:")
            for exc in raises:
                enhanced_lines.append(f"    {exc}: Description of when {exc} is raised")
            enhanced_lines.append("")

        # Wrap the result in triple quotes
        if enhanced_lines:
            result = '"""' + "\n".join(enhanced_lines) + '\n"""'
        else:
            result = '""""""'

        return result

    def validate_docstring_format(self, docstring: str) -> Dict[str, Any]:
        """
        Validate docstring format and provide detailed feedback.

        Args:
            docstring: The docstring to validate

        Returns:
            Dictionary with validation results
        """
        result = {
            "is_valid": True,
            "format": "unknown",
            "issues": [],
            "suggestions": [],
        }

        if not docstring:
            result["is_valid"] = False
            result["issues"].append("Missing docstring")
            result["suggestions"].append("Add comprehensive docstring")
            return result

        # Check format
        if self._is_google_format(docstring):
            result["format"] = "google"
        elif self._is_numpy_format(docstring):
            result["format"] = "numpy"
        else:
            result["format"] = "unknown"
            result["issues"].append("Unrecognized docstring format")
            result["suggestions"].append("Use Google or NumPy docstring format")

        # Check completeness
        completeness = self._check_docstring_completeness(docstring)
        if completeness < 0.7:
            result["issues"].append(".1f")
            result["suggestions"].append("Add missing sections: Args, Returns, Raises")

        # Check length
        if len(docstring.strip()) < 30:
            result["issues"].append("Docstring too short")
            result["suggestions"].append("Provide more detailed description")

        result["is_valid"] = len(result["issues"]) == 0
        return result


# Global instance
_docstring_standardizer = None


def get_docstring_standardizer() -> DocstringStandardizer:
    """Get the global docstring standardizer instance."""
    global _docstring_standardizer
    if _docstring_standardizer is None:
        _docstring_standardizer = DocstringStandardizer()
    return _docstring_standardizer


def analyze_documentation() -> Dict[str, Any]:
    """Convenience function to analyze codebase documentation."""
    standardizer = get_docstring_standardizer()
    return standardizer.analyze_codebase_documentation()


def standardize_docstring(docstring: str, **kwargs) -> str:
    """Convenience function to standardize a docstring."""
    standardizer = get_docstring_standardizer()
    return standardizer.standardize_docstring(docstring, **kwargs)


def validate_docstring(docstring: str) -> Dict[str, Any]:
    """Convenience function to validate a docstring."""
    standardizer = get_docstring_standardizer()
    return standardizer.validate_docstring_format(docstring)
