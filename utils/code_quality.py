"""
Code Quality Analysis and Refactoring Tools.

Provides cyclomatic complexity analysis, method extraction utilities,
and code quality assessment tools for maintaining high code standards.
"""

import ast
import inspect
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import radon.complexity as radon_cc
import radon.metrics as radon_metrics
from radon.visitors import ComplexityVisitor
import mccabe
import logging

logger = logging.getLogger(__name__)


class CodeComplexityAnalyzer:
    """
    Analyzes code complexity using multiple metrics and provides
    refactoring recommendations.
    """

    def __init__(self, max_complexity: int = 10, max_lines: int = 50):
        """Initialize the complexity analyzer.

        Args:
            max_complexity: Maximum allowed cyclomatic complexity
            max_lines: Maximum allowed lines per method
        """
        self.max_complexity = max_complexity
        self.max_lines = max_lines
        self.analysis_results: Dict[str, Any] = {}

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file for complexity issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=file_path)

            # Analyze complexity
            complexity_results = self._analyze_complexity(content)
            metrics_results = self._analyze_metrics(content)

            # Find complex methods
            complex_methods = self._find_complex_methods(tree, complexity_results)

            # Generate refactoring suggestions
            suggestions = self._generate_refactoring_suggestions(complex_methods, tree)

            result = {
                "file_path": file_path,
                "complexity_score": complexity_results,
                "metrics": metrics_results,
                "complex_methods": complex_methods,
                "refactoring_suggestions": suggestions,
                "overall_score": self._calculate_overall_score(complexity_results, metrics_results)
            }

            self.analysis_results[file_path] = result
            return result

        except Exception as e:
            logger.exception(f"Error analyzing file {file_path}: {e}")
            return {"error": str(e)}

    def analyze_directory(self, directory_path: str, file_pattern: str = "*.py") -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        directory = Path(directory_path)
        results = {}

        for file_path in directory.rglob(file_pattern):
            if file_path.is_file():
                results[str(file_path)] = self.analyze_file(str(file_path))

        return results

    def _analyze_complexity(self, content: str) -> Dict[str, Any]:
        """Analyze cyclomatic complexity using radon."""
        try:
            # Use radon to analyze complexity
            results = radon_cc.cc_visit(content)

            complexity_data = {}
            for result in results:
                key = f"{result.name}@{result.lineno}"
                complexity_data[key] = {
                    "name": result.name,
                    "complexity": result.complexity,
                    "line_number": result.lineno,
                    "classification": self._classify_complexity(result.complexity)
                }

            return complexity_data

        except Exception as e:
            logger.exception(f"Error in complexity analysis: {e}")
            return {}

    def _analyze_metrics(self, content: str) -> Dict[str, Any]:
        """Analyze code metrics using radon."""
        try:
            # Analyze maintainability index and other metrics
            mi = radon_metrics.mi_visit(content, multi=True)

            return {
                "maintainability_index": mi,
                "maintainability_grade": self._classify_maintainability(mi),
                "lines_of_code": len(content.split('\n')),
                "functions_count": len([line for line in content.split('\n') if line.strip().startswith('def ')]),
                "classes_count": len([line for line in content.split('\n') if line.strip().startswith('class ')])
            }

        except Exception as e:
            logger.exception(f"Error in metrics analysis: {e}")
            return {}

    def _find_complex_methods(self, tree: ast.AST, complexity_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find methods that exceed complexity thresholds."""
        complex_methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Find complexity for this method
                method_complexity = None
                for key, data in complexity_results.items():
                    if data["name"] == node.name:
                        method_complexity = data["complexity"]
                        break

                if method_complexity and method_complexity > self.max_complexity:
                    complex_methods.append({
                        "name": node.name,
                        "line_number": node.lineno,
                        "complexity": method_complexity,
                        "line_count": node.end_lineno - node.lineno if node.end_lineno else 0,
                        "classification": self._classify_complexity(method_complexity)
                    })

        return complex_methods

    def _classify_complexity(self, complexity: int) -> str:
        """Classify complexity level."""
        if complexity <= 5:
            return "simple"
        elif complexity <= 10:
            return "moderate"
        elif complexity <= 20:
            return "complex"
        elif complexity <= 30:
            return "very_complex"
        else:
            return "extremely_complex"

    def _classify_maintainability(self, mi: float) -> str:
        """Classify maintainability index."""
        if mi >= 20:
            return "excellent"
        elif mi >= 10:
            return "good"
        elif mi >= 0:
            return "fair"
        else:
            return "poor"

    def _calculate_overall_score(self, complexity_results: Dict[str, Any],
                               metrics_results: Dict[str, Any]) -> float:
        """Calculate overall code quality score."""
        score = 100.0

        # Complexity penalties
        high_complexity_count = sum(1 for data in complexity_results.values()
                                  if data["complexity"] > self.max_complexity)
        score -= high_complexity_count * 5

        # Maintainability penalties
        mi = metrics_results.get("maintainability_index", 20)
        if mi < 10:
            score -= (10 - mi) * 2

        # Size penalties
        loc = metrics_results.get("lines_of_code", 0)
        if loc > 1000:
            score -= (loc - 1000) / 100

        return max(0, score)

    def _generate_refactoring_suggestions(self, complex_methods: List[Dict[str, Any]],
                                        tree: ast.AST) -> List[Dict[str, Any]]:
        """Generate refactoring suggestions for complex methods."""
        suggestions = []

        for method in complex_methods:
            suggestion = {
                "method": method["name"],
                "complexity": method["complexity"],
                "line_number": method["line_number"],
                "suggestions": []
            }

            # Extract method suggestions
            if method["complexity"] > 15:
                suggestion["suggestions"].append({
                    "type": "extract_method",
                    "description": "Break down into smaller methods with single responsibilities",
                    "estimated_complexity_reduction": method["complexity"] - 8
                })

            # Conditional logic suggestions
            if method["complexity"] > 12:
                suggestion["suggestions"].append({
                    "type": "replace_conditional_with_polymorphism",
                    "description": "Replace complex conditional logic with polymorphism",
                    "estimated_complexity_reduction": 5
                })

            # Long method suggestions
            if method["line_count"] > self.max_lines:
                suggestion["suggestions"].append({
                    "type": "extract_method",
                    "description": f"Method is {method['line_count']} lines long, consider splitting",
                    "estimated_complexity_reduction": 3
                })

            suggestions.append(suggestion)

        return suggestions

    def generate_report(self, output_file: str = "complexity_report.md"):
        """Generate a comprehensive complexity report."""
        report = "# Code Complexity Analysis Report\n\n"

        for file_path, result in self.analysis_results.items():
            report += f"## {file_path}\n\n"

            if "error" in result:
                report += f"**Error:** {result['error']}\n\n"
                continue

            # Overall score
            score = result.get("overall_score", 0)
            report += f"**Overall Score:** {score:.1f}/100\n\n"

            # Complexity summary
            complexity_data = result.get("complexity_score", {})
            if complexity_data:
                report += "### Complexity Analysis\n\n"
                report += "| Method | Complexity | Classification |\n"
                report += "|--------|------------|----------------|\n"

                for key, data in complexity_data.items():
                    report += f"| {data['name']} | {data['complexity']} | {data['classification']} |\n"

                report += "\n"

            # Metrics
            metrics = result.get("metrics", {})
            if metrics:
                report += "### Code Metrics\n\n"
                report += f"- **Maintainability Index:** {metrics.get('maintainability_index', 'N/A')}\n"
                report += f"- **Maintainability Grade:** {metrics.get('maintainability_grade', 'N/A')}\n"
                report += f"- **Lines of Code:** {metrics.get('lines_of_code', 0)}\n"
                report += f"- **Functions:** {metrics.get('functions_count', 0)}\n"
                report += f"- **Classes:** {metrics.get('classes_count', 0)}\n\n"

            # Refactoring suggestions
            suggestions = result.get("refactoring_suggestions", [])
            if suggestions:
                report += "### Refactoring Suggestions\n\n"
                for suggestion in suggestions:
                    report += f"#### {suggestion['method']} (Complexity: {suggestion['complexity']})\n\n"
                    for sug in suggestion["suggestions"]:
                        report += f"- **{sug['type']}**: {sug['description']}\n"
                        if "estimated_complexity_reduction" in sug:
                            report += f"  - Estimated complexity reduction: {sug['estimated_complexity_reduction']}\n"
                    report += "\n"

        with open(output_file, "w") as f:
            f.write(report)

        logger.info(f"Complexity report generated: {output_file}")


class MethodExtractor:
    """
    Tool for extracting methods from complex functions to improve readability
    and maintainability.
    """

    def __init__(self):
        self.extracted_methods: List[Dict[str, Any]] = []

    def extract_method_from_function(self, source_code: str, function_name: str,
                                   start_line: int, end_line: int,
                                   new_method_name: str) -> str:
        """Extract a portion of a function into a new method."""
        lines = source_code.split('\n')

        # Extract the method portion
        extracted_lines = lines[start_line-1:end_line]

        # Create indentation for extracted method
        base_indent = self._get_base_indentation(extracted_lines[0])
        method_lines = [line[base_indent:] for line in extracted_lines]

        # Create new method
        new_method = f"""
    def {new_method_name}(self):
        \"\"\"Extracted method from {function_name}.\"\"\"
{chr(10).join('        ' + line for line in method_lines)}
"""

        # Replace original code with method call
        replacement = f"        self.{new_method_name}()"

        # Store extraction info
        self.extracted_methods.append({
            "original_function": function_name,
            "new_method": new_method_name,
            "extracted_lines": len(extracted_lines),
            "start_line": start_line,
            "end_line": end_line
        })

        return new_method, replacement

    def _get_base_indentation(self, line: str) -> int:
        """Get the base indentation of a line."""
        return len(line) - len(line.lstrip())

    def suggest_extractions(self, source_code: str, function_name: str) -> List[Dict[str, Any]]:
        """Suggest method extractions for a complex function."""
        suggestions = []

        # Parse the function
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Analyze the function body for extraction opportunities
                body_lines = len(node.body)

                if body_lines > 30:  # Long function
                    # Look for logical blocks that could be extracted
                    suggestions.extend(self._analyze_function_body(node))

                break

        return suggestions

    def _analyze_function_body(self, function_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Analyze function body for extraction opportunities."""
        suggestions = []

        # Look for repeated code patterns
        # Look for long conditional blocks
        # Look for complex expressions

        for i, node in enumerate(function_node.body):
            if isinstance(node, ast.If):
                # Check if the if block is long
                if_block_length = self._get_block_length(node.body)
                if if_block_length > 10:
                    suggestions.append({
                        "type": "extract_conditional",
                        "line_number": node.lineno,
                        "block_length": if_block_length,
                        "suggestion": f"Extract the if block starting at line {node.lineno} into a separate method"
                    })

            elif isinstance(node, ast.For) or isinstance(node, ast.While):
                # Check loop complexity
                loop_length = self._get_block_length(node.body)
                if loop_length > 15:
                    suggestions.append({
                        "type": "extract_loop",
                        "line_number": node.lineno,
                        "block_length": loop_length,
                        "suggestion": f"Extract the loop starting at line {node.lineno} into a separate method"
                    })

        return suggestions

    def _get_block_length(self, body: List[ast.stmt]) -> int:
        """Get the length of a code block."""
        if not body:
            return 0

        start_line = body[0].lineno
        end_line = body[-1].end_lineno if hasattr(body[-1], 'end_lineno') and body[-1].end_lineno else body[-1].lineno

        return end_line - start_line + 1


class CodeReviewChecklist:
    """
    Automated code review checklist generator and validator.
    """

    def __init__(self):
        self.checklist_items = {
            "complexity": {
                "name": "Cyclomatic Complexity",
                "description": "Functions should have complexity ≤ 10",
                "severity": "high",
                "automated": True
            },
            "line_length": {
                "name": "Line Length",
                "description": "Lines should be ≤ 100 characters",
                "severity": "low",
                "automated": True
            },
            "function_length": {
                "name": "Function Length",
                "description": "Functions should be ≤ 50 lines",
                "severity": "medium",
                "automated": True
            },
            "docstrings": {
                "name": "Documentation",
                "description": "All public functions should have docstrings",
                "severity": "medium",
                "automated": True
            },
            "error_handling": {
                "name": "Error Handling",
                "description": "Appropriate exception handling should be present",
                "severity": "high",
                "automated": False
            },
            "security": {
                "name": "Security",
                "description": "No hardcoded secrets or security vulnerabilities",
                "severity": "critical",
                "automated": False
            },
            "testing": {
                "name": "Test Coverage",
                "description": "Critical functions should have unit tests",
                "severity": "high",
                "automated": False
            },
            "naming": {
                "name": "Naming Conventions",
                "description": "Follow PEP 8 naming conventions",
                "severity": "low",
                "automated": True
            }
        }

    def generate_checklist(self, file_path: str) -> Dict[str, Any]:
        """Generate a code review checklist for a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            results = {}

            for item_key, item_config in self.checklist_items.items():
                if item_config["automated"]:
                    results[item_key] = self._run_automated_check(item_key, content)
                else:
                    results[item_key] = {
                        "status": "manual_review_required",
                        "description": item_config["description"]
                    }

            return {
                "file_path": file_path,
                "checklist": results,
                "overall_score": self._calculate_checklist_score(results)
            }

        except Exception as e:
            logger.exception(f"Error generating checklist for {file_path}: {e}")
            return {"error": str(e)}

    def _run_automated_check(self, check_type: str, content: str) -> Dict[str, Any]:
        """Run an automated check on the code."""
        if check_type == "complexity":
            return self._check_complexity(content)
        elif check_type == "line_length":
            return self._check_line_length(content)
        elif check_type == "function_length":
            return self._check_function_length(content)
        elif check_type == "docstrings":
            return self._check_docstrings(content)
        elif check_type == "naming":
            return self._check_naming(content)
        else:
            return {"status": "not_implemented"}

    def _check_complexity(self, content: str) -> Dict[str, Any]:
        """Check cyclomatic complexity."""
        try:
            results = radon_cc.cc_visit(content)
            high_complexity = [r for r in results if r.complexity > 10]

            return {
                "status": "pass" if not high_complexity else "fail",
                "details": f"{len(high_complexity)} functions with complexity > 10",
                "functions": [r.name for r in high_complexity]
            }
        except Exception:
            return {"status": "error", "details": "Could not analyze complexity"}

    def _check_line_length(self, content: str) -> Dict[str, Any]:
        """Check line lengths."""
        lines = content.split('\n')
        long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]

        return {
            "status": "pass" if not long_lines else "fail",
            "details": f"{len(long_lines)} lines exceed 100 characters",
            "line_numbers": long_lines[:10]  # Show first 10
        }

    def _check_function_length(self, content: str) -> Dict[str, Any]:
        """Check function lengths."""
        tree = ast.parse(content)
        long_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                length = node.end_lineno - node.lineno if node.end_lineno else 0
                if length > 50:
                    long_functions.append({
                        "name": node.name,
                        "length": length,
                        "line_number": node.lineno
                    })

        return {
            "status": "pass" if not long_functions else "fail",
            "details": f"{len(long_functions)} functions exceed 50 lines",
            "functions": long_functions
        }

    def _check_docstrings(self, content: str) -> Dict[str, Any]:
        """Check for missing docstrings."""
        tree = ast.parse(content)
        missing_docstrings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                # Check if function has a docstring
                if not (node.body and isinstance(node.body[0], ast.Expr) and
                       isinstance(node.body[0].value, ast.Str)):
                    missing_docstrings.append(node.name)

        return {
            "status": "pass" if not missing_docstrings else "fail",
            "details": f"{len(missing_docstrings)} public functions missing docstrings",
            "functions": missing_docstrings[:10]  # Show first 10
        }

    def _check_naming(self, content: str) -> Dict[str, Any]:
        """Check naming conventions."""
        # This is a simplified check - could be enhanced with more sophisticated analysis
        issues = []

        # Check for camelCase in function names (should be snake_case)
        camel_case_pattern = r'\bdef\s+[a-z]+[A-Z][a-zA-Z]*\b'
        matches = re.findall(camel_case_pattern, content)
        if matches:
            issues.extend(matches)

        return {
            "status": "pass" if not issues else "fail",
            "details": f"{len(issues)} potential naming convention issues",
            "issues": issues[:5]  # Show first 5
        }

    def _calculate_checklist_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall checklist score."""
        total_score = 0
        max_score = 0

        severity_weights = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4
        }

        for item_key, result in results.items():
            item_config = self.checklist_items[item_key]
            weight = severity_weights.get(item_config["severity"], 1)
            max_score += weight

            if result.get("status") == "pass":
                total_score += weight
            elif result.get("status") == "manual_review_required":
                total_score += weight * 0.5  # Partial credit for manual review items

        return (total_score / max_score * 100) if max_score > 0 else 0

    def generate_report(self, file_path: str, output_file: str = "code_review_report.md"):
        """Generate a code review report."""
        checklist_result = self.generate_checklist(file_path)

        if "error" in checklist_result:
            report = f"# Code Review Report\n\n**Error:** {checklist_result['error']}\n"
        else:
            report = f"# Code Review Report for {file_path}\n\n"
            report += f"**Overall Score:** {checklist_result['overall_score']:.1f}/100\n\n"

            report += "## Checklist Results\n\n"

            for item_key, result in checklist_result["checklist"].items():
                item_config = self.checklist_items[item_key]
                status = result.get("status", "unknown")
                status_icon = "✅" if status == "pass" else "❌" if status == "fail" else "🔍"

                report += f"### {status_icon} {item_config['name']}\n\n"
                report += f"**Severity:** {item_config['severity'].title()}\n\n"
                report += f"**Description:** {item_config['description']}\n\n"
                report += f"**Status:** {status.title()}\n\n"

                if "details" in result:
                    report += f"**Details:** {result['details']}\n\n"

                if "functions" in result and result["functions"]:
                    report += "**Affected Functions:**\n"
                    for func in result["functions"][:5]:  # Show first 5
                        if isinstance(func, dict):
                            report += f"- {func.get('name', 'Unknown')} (line {func.get('line_number', '?')})\n"
                        else:
                            report += f"- {func}\n"
                    if len(result["functions"]) > 5:
                        report += f"- ... and {len(result['functions']) - 5} more\n"
                    report += "\n"

        with open(output_file, "w") as f:
            f.write(report)

        logger.info(f"Code review report generated: {output_file}")


# Utility functions
def analyze_code_quality(directory_path: str) -> Dict[str, Any]:
    """Analyze code quality for an entire directory."""
    analyzer = CodeComplexityAnalyzer()
    checklist = CodeReviewChecklist()

    results = analyzer.analyze_directory(directory_path)

    # Generate summary
    summary = {
        "total_files": len(results),
        "complexity_issues": 0,
        "quality_score": 0,
        "files_analyzed": []
    }

    for file_path, result in results.items():
        if "error" not in result:
            summary["files_analyzed"].append({
                "path": file_path,
                "score": result.get("overall_score", 0),
                "complex_methods": len(result.get("complex_methods", []))
            })
            summary["complexity_issues"] += len(result.get("complex_methods", []))
            summary["quality_score"] += result.get("overall_score", 0)

    if summary["total_files"] > 0:
        summary["quality_score"] /= summary["total_files"]

    return summary


def generate_quality_report(directory_path: str, output_dir: str = "reports"):
    """Generate comprehensive code quality reports."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Analyze complexity
    analyzer = CodeComplexityAnalyzer()
    analyzer.analyze_directory(directory_path)
    analyzer.generate_report(output_path / "complexity_report.md")

    # Generate code review reports for each file
    checklist = CodeReviewChecklist()

    for file_path in Path(directory_path).rglob("*.py"):
        if file_path.is_file():
            report_name = f"review_{file_path.name.replace('.py', '')}.md"
            checklist.generate_report(str(file_path), output_path / report_name)

    # Generate summary
    summary = analyze_code_quality(directory_path)

    summary_report = f"""# Code Quality Summary Report

## Overview
- **Total Files Analyzed:** {summary['total_files']}
- **Average Quality Score:** {summary['quality_score']:.1f}/100
- **Total Complexity Issues:** {summary['complexity_issues']}

## File Details

| File | Quality Score | Complexity Issues |
|------|---------------|-------------------|
"""

    for file_info in summary["files_analyzed"]:
        summary_report += f"| {file_info['path']} | {file_info['score']:.1f} | {file_info['complex_methods']} |\n"

    summary_report += "\n## Recommendations\n\n"

    if summary["quality_score"] < 70:
        summary_report += "- **Critical:** Overall code quality needs improvement\n"
    if summary["complexity_issues"] > 0:
        summary_report += f"- **High:** {summary['complexity_issues']} methods exceed complexity thresholds\n"
    if summary["quality_score"] >= 85:
        summary_report += "- **Good:** Code quality standards are being maintained\n"

    with open(output_path / "quality_summary.md", "w") as f:
        f.write(summary_report)

    logger.info(f"Quality reports generated in {output_path}")
