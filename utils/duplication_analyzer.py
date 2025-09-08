"""
Code Duplication Analyzer - Automated code similarity analysis and refactoring.

Provides code similarity detection, shared utility libraries, and duplication prevention
for maintaining clean, DRY (Don't Repeat Yourself) codebase.
"""

import ast
import logging
import re
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from difflib import SequenceMatcher
import hashlib
from collections import defaultdict
import inspect

from utils.constants import PROJECT_ROOT
from utils.error_handler import ErrorHandler, TradingError

logger = logging.getLogger(__name__)


class CodeBlock:
    """Represents a block of code for similarity analysis."""

    def __init__(self, content: str, file_path: str, start_line: int, end_line: int,
                 block_type: str = "function"):
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.block_type = block_type
        self.hash = self._calculate_hash()
        self.normalized_content = self._normalize_content()

    def _calculate_hash(self) -> str:
        """Calculate hash of the code block."""
        return hashlib.md5(self.content.encode()).hexdigest()

    def _normalize_content(self) -> str:
        """Normalize content for better similarity comparison."""
        # Remove comments
        content = re.sub(r'#.*$', '', self.content, flags=re.MULTILINE)
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        # Remove variable names (replace with placeholders)
        content = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', content)
        return content

    def similarity_to(self, other: 'CodeBlock') -> float:
        """Calculate similarity to another code block."""
        return SequenceMatcher(None, self.normalized_content, other.normalized_content).ratio()

    def __str__(self) -> str:
        return f"{self.block_type} in {self.file_path}:{self.start_line}-{self.end_line}"


class CodeDuplicationAnalyzer:
    """
    Analyzes code for duplication and suggests refactoring opportunities.
    """

    def __init__(self, min_similarity: float = 0.8, min_lines: int = 5):
        self.min_similarity = min_similarity
        self.min_lines = min_lines
        self.code_blocks: List[CodeBlock] = []
        self.duplicate_groups: List[List[CodeBlock]] = []
        self.shared_utilities: Dict[str, List[str]] = {}

    def analyze_directory(self, directory_path: str, file_pattern: str = "*.py") -> Dict[str, Any]:
        """Analyze directory for code duplication."""
        logger.info(f"Starting code duplication analysis in {directory_path}")

        directory = Path(directory_path)

        # Extract code blocks from all files
        for file_path in directory.rglob(file_pattern):
            if file_path.is_file():
                self._extract_code_blocks(str(file_path))

        # Find duplicates
        self._find_duplicates()

        # Generate refactoring suggestions
        suggestions = self._generate_refactoring_suggestions()

        return {
            "total_files_analyzed": len(list(directory.rglob(file_pattern))),
            "total_code_blocks": len(self.code_blocks),
            "duplicate_groups_found": len(self.duplicate_groups),
            "total_duplicated_lines": self._calculate_total_duplicated_lines(),
            "duplication_percentage": self._calculate_duplication_percentage(),
            "refactoring_suggestions": suggestions,
            "shared_utility_candidates": self._identify_shared_utility_candidates()
        }

    def _extract_code_blocks(self, file_path: str):
        """Extract code blocks from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function code
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', start_line + 10)
                    lines = content.split('\n')

                    if end_line - start_line >= self.min_lines:
                        block_content = '\n'.join(lines[start_line-1:end_line])
                        code_block = CodeBlock(
                            block_content, file_path, start_line, end_line, "function"
                        )
                        self.code_blocks.append(code_block)

                elif isinstance(node, ast.ClassDef):
                    # Extract class methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            start_line = item.lineno
                            end_line = getattr(item, 'end_lineno', start_line + 10)
                            lines = content.split('\n')

                            if end_line - start_line >= self.min_lines:
                                block_content = '\n'.join(lines[start_line-1:end_line])
                                code_block = CodeBlock(
                                    block_content, file_path, start_line, end_line,
                                    f"method_{item.name}"
                                )
                                self.code_blocks.append(code_block)

        except Exception as e:
            logger.warning(f"Error extracting code blocks from {file_path}: {e}")

    def _find_duplicates(self):
        """Find duplicate code blocks."""
        processed = set()

        for i, block1 in enumerate(self.code_blocks):
            if block1.hash in processed:
                continue

            duplicates = [block1]

            for j, block2 in enumerate(self.code_blocks):
                if i != j and block2.hash not in processed:
                    if block1.similarity_to(block2) >= self.min_similarity:
                        duplicates.append(block2)
                        processed.add(block2.hash)

            if len(duplicates) > 1:
                self.duplicate_groups.append(duplicates)

            processed.add(block1.hash)

    def _calculate_total_duplicated_lines(self) -> int:
        """Calculate total lines of duplicated code."""
        total_lines = 0
        processed_blocks = set()

        for group in self.duplicate_groups:
            for block in group[1:]:  # Skip the first occurrence
                if block.hash not in processed_blocks:
                    total_lines += (block.end_line - block.start_line + 1)
                    processed_blocks.add(block.hash)

        return total_lines

    def _calculate_duplication_percentage(self) -> float:
        """Calculate percentage of duplicated code."""
        total_lines = sum(block.end_line - block.start_line + 1 for block in self.code_blocks)
        duplicated_lines = self._calculate_total_duplicated_lines()

        return (duplicated_lines / total_lines * 100) if total_lines > 0 else 0

    def _generate_refactoring_suggestions(self) -> List[Dict[str, Any]]:
        """Generate refactoring suggestions for duplicated code."""
        suggestions = []

        for i, group in enumerate(self.duplicate_groups):
            if len(group) < 2:
                continue

            # Calculate group metrics
            avg_lines = sum(block.end_line - block.start_line + 1 for block in group) / len(group)
            files = list(set(block.file_path for block in group))

            suggestion = {
                "group_id": i + 1,
                "duplicate_count": len(group),
                "average_lines": int(avg_lines),
                "affected_files": files,
                "locations": [
                    {
                        "file": block.file_path,
                        "start_line": block.start_line,
                        "end_line": block.end_line,
                        "block_type": block.block_type
                    }
                    for block in group
                ],
                "refactoring_type": self._determine_refactoring_type(group),
                "estimated_savings": int(avg_lines * (len(group) - 1)),
                "priority": self._calculate_refactoring_priority(group)
            }

            suggestions.append(suggestion)

        # Sort by priority
        suggestions.sort(key=lambda x: x["priority"], reverse=True)
        return suggestions

    def _determine_refactoring_type(self, group: List[CodeBlock]) -> str:
        """Determine the best refactoring approach for a duplicate group."""
        block_types = [block.block_type for block in group]

        if all("function" in bt for bt in block_types):
            return "extract_shared_function"
        elif all("method" in bt for bt in block_types):
            return "extract_shared_method"
        else:
            return "extract_utility_function"

    def _calculate_refactoring_priority(self, group: List[CodeBlock]) -> int:
        """Calculate refactoring priority based on various factors."""
        priority = 0

        # More duplicates = higher priority
        priority += len(group) * 10

        # Longer code blocks = higher priority
        avg_lines = sum(block.end_line - block.start_line + 1 for block in group) / len(group)
        priority += min(avg_lines, 50)  # Cap at 50

        # Cross-file duplicates = higher priority
        files = set(block.file_path for block in group)
        if len(files) > 1:
            priority += 20

        return priority

    def _identify_shared_utility_candidates(self) -> List[Dict[str, Any]]:
        """Identify candidates for shared utility functions."""
        candidates = []

        # Group by functionality patterns
        pattern_groups = defaultdict(list)

        for block in self.code_blocks:
            # Simple pattern recognition
            if "calculate" in block.content.lower():
                pattern_groups["calculation"].append(block)
            elif "validate" in block.content.lower():
                pattern_groups["validation"].append(block)
            elif "format" in block.content.lower():
                pattern_groups["formatting"].append(block)
            elif "parse" in block.content.lower():
                pattern_groups["parsing"].append(block)

        for pattern, blocks in pattern_groups.items():
            if len(blocks) >= 3:  # At least 3 occurrences
                candidates.append({
                    "pattern": pattern,
                    "occurrences": len(blocks),
                    "affected_files": list(set(b.file_path for b in blocks)),
                    "suggested_utility": f"create_shared_{pattern}_utility"
                })

        return candidates

    def generate_duplication_report(self, output_file: str = "duplication_report.md"):
        """Generate comprehensive duplication analysis report."""
        report = "# Code Duplication Analysis Report\n\n"

        if not self.duplicate_groups:
            report += "✅ No significant code duplication found.\n\n"
            return report

        # Summary
        total_duplicated_lines = self._calculate_total_duplicated_lines()
        duplication_percentage = self._calculate_duplication_percentage()

        report += "## Summary\n\n"
        report += f"- **Duplicate Groups Found:** {len(self.duplicate_groups)}\n"
        report += f"- **Total Duplicated Lines:** {total_duplicated_lines}\n"
        report += f"- **Duplication Percentage:** {duplication_percentage:.1f}%\n"
        report += f"- **Code Blocks Analyzed:** {len(self.code_blocks)}\n\n"

        # Detailed findings
        report += "## Duplicate Code Groups\n\n"

        for i, group in enumerate(self.duplicate_groups):
            report += f"### Group {i + 1}\n\n"
            report += f"- **Occurrences:** {len(group)}\n"
            report += f"- **Average Lines:** {sum(b.end_line - b.start_line + 1 for b in group) // len(group)}\n"
            report += f"- **Refactoring Priority:** {self._calculate_refactoring_priority(group)}\n\n"

            report += "#### Locations:\n\n"
            for block in group:
                report += f"- `{block.file_path}:{block.start_line}-{block.end_line}` ({block.block_type})\n"

            refactoring_type = self._determine_refactoring_type(group)
            report += f"\n**Suggested Refactoring:** {refactoring_type}\n\n"

        # Recommendations
        report += "## Recommendations\n\n"

        if duplication_percentage > 20:
            report += "🚨 **HIGH PRIORITY:** Significant code duplication detected. Immediate refactoring recommended.\n\n"
        elif duplication_percentage > 10:
            report += "⚠️ **MEDIUM PRIORITY:** Moderate code duplication found. Consider refactoring.\n\n"
        else:
            report += "ℹ️ **LOW PRIORITY:** Minor code duplication detected. Address during regular maintenance.\n\n"

        # Specific recommendations
        high_priority_groups = [g for g in self.duplicate_groups if self._calculate_refactoring_priority(g) > 50]
        if high_priority_groups:
            report += "### High Priority Refactoring Targets:\n\n"
            for i, group in enumerate(high_priority_groups[:5]):  # Top 5
                report += f"{i+1}. Group {self.duplicate_groups.index(group) + 1} "
                report += f"({len(group)} occurrences, "
                report += f"{sum(b.end_line - b.start_line + 1 for b in group) // len(group)} avg lines)\n"

        with open(output_file, "w") as f:
            f.write(report)

        logger.info(f"Duplication report generated: {output_file}")


class SharedUtilityLibrary:
    """
    Manages shared utility functions and prevents code duplication.
    """

    def __init__(self):
        self.utilities: Dict[str, Dict[str, Any]] = {}
        self.usage_tracking: Dict[str, List[str]] = defaultdict(list)

    def add_utility_function(self, name: str, function: callable,
                           description: str = "", category: str = "general"):
        """Add a shared utility function."""
        self.utilities[name] = {
            "function": function,
            "description": description,
            "category": category,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }

        logger.info(f"Added shared utility: {name} ({category})")

    def get_utility(self, name: str) -> Optional[callable]:
        """Get a shared utility function."""
        if name in self.utilities:
            self.utilities[name]["usage_count"] += 1
            return self.utilities[name]["function"]
        return None

    def track_usage(self, utility_name: str, file_path: str, line_number: int):
        """Track usage of a utility function."""
        self.usage_tracking[utility_name].append(f"{file_path}:{line_number}")

    def get_utilities_report(self) -> Dict[str, Any]:
        """Generate report on shared utilities."""
        return {
            "total_utilities": len(self.utilities),
            "categories": self._get_category_breakdown(),
            "most_used": self._get_most_used_utilities(),
            "usage_tracking": dict(self.usage_tracking)
        }

    def _get_category_breakdown(self) -> Dict[str, int]:
        """Get breakdown of utilities by category."""
        breakdown = defaultdict(int)
        for utility in self.utilities.values():
            breakdown[utility["category"]] += 1
        return dict(breakdown)

    def _get_most_used_utilities(self) -> List[Dict[str, Any]]:
        """Get most used utilities."""
        utilities_list = [
            {"name": name, "usage_count": data["usage_count"], "category": data["category"]}
            for name, data in self.utilities.items()
        ]

        utilities_list.sort(key=lambda x: x["usage_count"], reverse=True)
        return utilities_list[:10]  # Top 10


class CodeDuplicationManager:
    """
    Comprehensive code duplication management system.
    """

    def __init__(self):
        self.analyzer = CodeDuplicationAnalyzer()
        self.utility_library = SharedUtilityLibrary()
        self.refactoring_history: List[Dict[str, Any]] = []

    async def perform_duplication_analysis(self, directory_path: str = ".") -> Dict[str, Any]:
        """Perform comprehensive code duplication analysis."""
        logger.info("Starting comprehensive code duplication analysis")

        # Analyze for duplication
        analysis_results = self.analyzer.analyze_directory(directory_path)

        # Generate recommendations
        recommendations = self._generate_duplication_recommendations(analysis_results)

        # Create shared utilities suggestions
        shared_utility_suggestions = self._suggest_shared_utilities(analysis_results)

        return {
            "analysis_results": analysis_results,
            "recommendations": recommendations,
            "shared_utility_suggestions": shared_utility_suggestions,
            "refactoring_roadmap": self._create_refactoring_roadmap(analysis_results)
        }

    def _generate_duplication_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing code duplication."""
        recommendations = []

        duplication_percentage = analysis_results.get("duplication_percentage", 0)
        duplicate_groups = analysis_results.get("duplicate_groups_found", 0)

        if duplication_percentage > 20:
            recommendations.append("🚨 CRITICAL: High code duplication detected - immediate refactoring required")
        elif duplication_percentage > 10:
            recommendations.append("⚠️ HIGH: Moderate code duplication found - prioritize refactoring")
        elif duplication_percentage > 5:
            recommendations.append("ℹ️ MEDIUM: Some code duplication detected - address in next sprint")

        if duplicate_groups > 10:
            recommendations.append(f"📊 Create {duplicate_groups} shared utility functions")
        elif duplicate_groups > 5:
            recommendations.append(f"🔧 Extract {duplicate_groups} common functions")

        # Specific recommendations based on patterns
        shared_candidates = analysis_results.get("shared_utility_candidates", [])
        if shared_candidates:
            recommendations.append(f"🏗️ Create shared utilities for: {', '.join(c['pattern'] for c in shared_candidates[:3])}")

        return recommendations

    def _suggest_shared_utilities(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest shared utilities based on duplication patterns."""
        suggestions = []

        candidates = analysis_results.get("shared_utility_candidates", [])
        refactoring_suggestions = analysis_results.get("refactoring_suggestions", [])

        # Combine and prioritize suggestions
        for candidate in candidates:
            suggestions.append({
                "type": "shared_utility",
                "pattern": candidate["pattern"],
                "occurrences": candidate["occurrences"],
                "affected_files": candidate["affected_files"],
                "suggested_name": candidate["suggested_utility"],
                "priority": "high" if candidate["occurrences"] > 5 else "medium"
            })

        for suggestion in refactoring_suggestions[:5]:  # Top 5
            suggestions.append({
                "type": "refactoring",
                "group_id": suggestion["group_id"],
                "occurrences": suggestion["duplicate_count"],
                "estimated_savings": suggestion["estimated_savings"],
                "refactoring_type": suggestion["refactoring_type"],
                "priority": "high" if suggestion["priority"] > 50 else "medium"
            })

        return suggestions

    def _create_refactoring_roadmap(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a roadmap for addressing code duplication."""
        roadmap = []

        # Phase 1: High-impact refactoring
        high_priority = [
            s for s in analysis_results.get("refactoring_suggestions", [])
            if s["priority"] > 60
        ]

        if high_priority:
            roadmap.append({
                "phase": 1,
                "name": "High-Impact Refactoring",
                "duration": "1-2 weeks",
                "tasks": [f"Refactor Group {s['group_id']} ({s['duplicate_count']} duplicates)" for s in high_priority[:5]],
                "estimated_savings": sum(s["estimated_savings"] for s in high_priority)
            })

        # Phase 2: Shared utilities creation
        shared_candidates = analysis_results.get("shared_utility_candidates", [])
        if shared_candidates:
            roadmap.append({
                "phase": 2,
                "name": "Shared Utilities Creation",
                "duration": "1 week",
                "tasks": [f"Create {c['suggested_utility']}" for c in shared_candidates],
                "estimated_savings": len(shared_candidates) * 50  # Rough estimate
            })

        # Phase 3: Cleanup and optimization
        medium_priority = [
            s for s in analysis_results.get("refactoring_suggestions", [])
            if 30 <= s["priority"] <= 60
        ]

        if medium_priority:
            roadmap.append({
                "phase": 3,
                "name": "Cleanup and Optimization",
                "duration": "2-3 weeks",
                "tasks": [f"Clean up Group {s['group_id']}" for s in medium_priority],
                "estimated_savings": sum(s["estimated_savings"] for s in medium_priority)
            })

        return roadmap

    def generate_duplication_management_report(self, output_file: str = "duplication_management_report.md"):
        """Generate comprehensive duplication management report."""
        report = "# Code Duplication Management Report\n\n"

        # Run analysis
        analysis_results = self.analyzer.analyze_directory(".")

        report += "## Current State\n\n"
        report += f"- **Duplication Percentage:** {analysis_results.get('duplication_percentage', 0):.1f}%\n"
        report += f"- **Duplicate Groups:** {analysis_results.get('duplicate_groups_found', 0)}\n"
        report += f"- **Duplicated Lines:** {analysis_results.get('total_duplicated_lines', 0)}\n\n"

        # Recommendations
        recommendations = self._generate_duplication_recommendations(analysis_results)
        if recommendations:
            report += "## Recommendations\n\n"
            for rec in recommendations:
                report += f"- {rec}\n"
            report += "\n"

        # Refactoring Roadmap
        roadmap = self._create_refactoring_roadmap(analysis_results)
        if roadmap:
            report += "## Refactoring Roadmap\n\n"
            for phase in roadmap:
                report += f"### Phase {phase['phase']}: {phase['name']}\n\n"
                report += f"**Duration:** {phase['duration']}\n\n"
                report += f"**Estimated Savings:** {phase.get('estimated_savings', 0)} lines\n\n"
                report += "**Tasks:**\n\n"
                for task in phase["tasks"]:
                    report += f"- {task}\n"
                report += "\n"

        # Shared Utilities Status
        utilities_report = self.utility_library.get_utilities_report()
        if utilities_report["total_utilities"] > 0:
            report += "## Shared Utilities Status\n\n"
            report += f"- **Total Utilities:** {utilities_report['total_utilities']}\n"
            report += f"- **Categories:** {utilities_report['categories']}\n\n"

            most_used = utilities_report.get("most_used", [])
            if most_used:
                report += "### Most Used Utilities\n\n"
                for utility in most_used[:5]:
                    report += f"- **{utility['name']}**: {utility['usage_count']} uses ({utility['category']})\n"
                report += "\n"

        with open(output_file, "w") as f:
            f.write(report)

        logger.info(f"Duplication management report generated: {output_file}")


# Global duplication manager instance
_duplication_manager = None

def get_duplication_manager() -> CodeDuplicationManager:
    """Get the global code duplication manager instance."""
    global _duplication_manager
    if _duplication_manager is None:
        _duplication_manager = CodeDuplicationManager()
    return _duplication_manager


# Utility functions
def analyze_code_duplication(directory_path: str = ".") -> Dict[str, Any]:
    """Convenience function to analyze code duplication."""
    manager = get_duplication_manager()
    return manager.analyzer.analyze_directory(directory_path)


def generate_duplication_report(output_file: str = "duplication_report.md"):
    """Convenience function to generate duplication report."""
    manager = get_duplication_manager()
    manager.analyzer.generate_duplication_report(output_file)


def get_shared_utility(name: str) -> Optional[callable]:
    """Convenience function to get a shared utility."""
    manager = get_duplication_manager()
    return manager.utility_library.get_utility(name)


def add_shared_utility(name: str, function: callable, description: str = "", category: str = "general"):
    """Convenience function to add a shared utility."""
    manager = get_duplication_manager()
    manager.utility_library.add_utility_function(name, function, description, category)
