"""Severity Scoring - Convert audit results to 0-100 health score with policy enforcement"""

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SeverityLevel(Enum):
    """Severity levels for code quality scores"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ScoreComponent:
    """Individual score component with weight and value"""

    name: str
    weight: float  # 0.0 to 1.0
    raw_score: float  # 0.0 to 1.0 (higher is better)
    weighted_score: float  # calculated
    issues_count: int
    description: str


@dataclass
class HealthScore:
    """Complete health score with breakdown and recommendations"""

    overall_score: float  # 0-100
    severity_level: SeverityLevel
    components: List[ScoreComponent]
    recommendations: List[str]
    policy_actions: List[str]
    breakdown: Dict[str, Any]


class SeverityScorer:
    """Converts audit results to health scores with policy enforcement"""

    # Default weights (total = 1.0)
    DEFAULT_WEIGHTS = {
        "security": 0.30,  # 30% - bandit security issues
        "lint": 0.20,  # 20% - linting errors
        "complexity": 0.15,  # 15% - code complexity
        "dead_code": 0.10,  # 10% - dead code detection
        "duplication": 0.10,  # 10% - code duplication (placeholder)
        "coverage": 0.15,  # 15% - test coverage gap
    }

    # Policy thresholds
    POLICY_THRESHOLDS = {
        "critical": 90,  # >= 90: block merges, notify maintainers
        "high": 75,  # 75-89: PR comments, create issues
        "medium": 50,  # 50-74: warnings, non-blocking
        "low": 0,  # < 50: informational
    }

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        self.weights = custom_weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Validate that weights sum to 1.0"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def calculate_health_score(self, audit_results: Dict[str, Any]) -> HealthScore:
        """Calculate comprehensive health score from audit results"""

        # Calculate individual component scores
        components = self._calculate_component_scores(audit_results)

        # Calculate weighted overall score
        overall_score = sum(comp.weighted_score for comp in components)

        # Determine severity level
        severity_level = self._determine_severity_level(overall_score)

        # Generate recommendations and policy actions
        recommendations = self._generate_recommendations(components, severity_level)
        policy_actions = self._generate_policy_actions(severity_level)

        # Create detailed breakdown
        breakdown = self._create_breakdown(components, audit_results)

        return HealthScore(
            overall_score=round(overall_score, 1),
            severity_level=severity_level,
            components=components,
            recommendations=recommendations,
            policy_actions=policy_actions,
            breakdown=breakdown,
        )

    def _calculate_component_scores(
        self, audit_results: Dict[str, Any]
    ) -> List[ScoreComponent]:
        """Calculate scores for each component"""
        components = []

        # Security score (bandit)
        security_score, security_issues = self._calculate_security_score(audit_results)
        components.append(
            ScoreComponent(
                name="security",
                weight=self.weights["security"],
                raw_score=security_score,
                weighted_score=security_score * self.weights["security"],
                issues_count=security_issues,
                description=f"Security issues: {security_issues} found",
            )
        )

        # Lint score
        lint_score, lint_issues = self._calculate_lint_score(audit_results)
        components.append(
            ScoreComponent(
                name="lint",
                weight=self.weights["lint"],
                raw_score=lint_score,
                weighted_score=lint_score * self.weights["lint"],
                issues_count=lint_issues,
                description=f"Linting issues: {lint_issues} found",
            )
        )

        # Complexity score
        complexity_score, complexity_issues = self._calculate_complexity_score(
            audit_results
        )
        components.append(
            ScoreComponent(
                name="complexity",
                weight=self.weights["complexity"],
                raw_score=complexity_score,
                weighted_score=complexity_score * self.weights["complexity"],
                issues_count=complexity_issues,
                description=f"Complex functions: {complexity_issues} found",
            )
        )

        # Dead code score
        dead_code_score, dead_code_issues = self._calculate_dead_code_score(
            audit_results
        )
        components.append(
            ScoreComponent(
                name="dead_code",
                weight=self.weights["dead_code"],
                raw_score=dead_code_score,
                weighted_score=dead_code_score * self.weights["dead_code"],
                issues_count=dead_code_issues,
                description=f"Dead code issues: {dead_code_issues} found",
            )
        )

        # Duplication score (placeholder - would need duplication analysis)
        duplication_score, duplication_issues = self._calculate_duplication_score(
            audit_results
        )
        components.append(
            ScoreComponent(
                name="duplication",
                weight=self.weights["duplication"],
                raw_score=duplication_score,
                weighted_score=duplication_score * self.weights["duplication"],
                issues_count=duplication_issues,
                description=f"Duplication issues: {duplication_issues} found",
            )
        )

        # Coverage score
        coverage_score, coverage_gap = self._calculate_coverage_score(audit_results)
        components.append(
            ScoreComponent(
                name="coverage",
                weight=self.weights["coverage"],
                raw_score=coverage_score,
                weighted_score=coverage_score * self.weights["coverage"],
                issues_count=coverage_gap,
                description=f"Coverage gap: {coverage_gap}%",
            )
        )

        return components

    def _calculate_security_score(
        self, audit_results: Dict[str, Any]
    ) -> Tuple[float, int]:
        """Calculate security score from bandit results"""
        security_data = audit_results.get("static_analysis", {}).get(
            "security_issues", {}
        )
        issues_count = security_data.get("summary", {}).get("total_issues", 0)

        # Security score: 1.0 for 0 issues, decreasing with more issues
        if issues_count == 0:
            score = 1.0
        elif issues_count <= 5:
            score = 0.8  # Minor issues
        elif issues_count <= 15:
            score = 0.6  # Moderate issues
        elif issues_count <= 30:
            score = 0.3  # Major issues
        else:
            score = 0.1  # Critical issues

        return score, issues_count

    def _calculate_lint_score(self, audit_results: Dict[str, Any]) -> Tuple[float, int]:
        """Calculate lint score from linting results"""
        lint_data = audit_results.get("linting", {})
        summary = lint_data.get("summary", {})
        failed_tools = summary.get("failed", 0)
        error_tools = summary.get("errors", 0)
        total_issues = failed_tools + error_tools

        # Lint score: 1.0 for all passing, decreasing with failures
        if total_issues == 0:
            score = 1.0
        elif total_issues == 1:
            score = 0.9  # One tool failed
        elif total_issues == 2:
            score = 0.7  # Two tools failed
        else:
            score = 0.4  # Multiple tools failed

        return score, total_issues

    def _calculate_complexity_score(
        self, audit_results: Dict[str, Any]
    ) -> Tuple[float, int]:
        """Calculate complexity score from radon results"""
        complexity_data = audit_results.get("static_analysis", {}).get(
            "complexity_issues", {}
        )
        distribution = complexity_data.get("complexity_distribution", {})

        very_complex = distribution.get("very_complex", 0)  # > 20 complexity
        complex_funcs = distribution.get("complex", 0)  # 11-20 complexity
        total_functions = complexity_data.get(
            "total_functions", 1
        )  # Avoid division by zero

        # Calculate complexity penalty
        complexity_ratio = (very_complex * 2 + complex_funcs) / max(total_functions, 1)

        # Complexity score: 1.0 for low complexity, decreasing with high complexity
        if complexity_ratio <= 0.05:
            score = 1.0  # Excellent
        elif complexity_ratio <= 0.15:
            score = 0.8  # Good
        elif complexity_ratio <= 0.30:
            score = 0.6  # Moderate
        elif complexity_ratio <= 0.50:
            score = 0.3  # High
        else:
            score = 0.1  # Very high

        return score, very_complex + complex_funcs

    def _calculate_dead_code_score(
        self, audit_results: Dict[str, Any]
    ) -> Tuple[float, int]:
        """Calculate dead code score from vulture results"""
        dead_code_data = audit_results.get("static_analysis", {}).get("dead_code", {})
        dead_code_count = dead_code_data.get("summary", {}).get(
            "potential_dead_code", 0
        )

        # Dead code score: 1.0 for no dead code, decreasing with more dead code
        if dead_code_count == 0:
            score = 1.0
        elif dead_code_count <= 5:
            score = 0.9  # Minor dead code
        elif dead_code_count <= 15:
            score = 0.7  # Moderate dead code
        elif dead_code_count <= 30:
            score = 0.4  # Significant dead code
        else:
            score = 0.2  # Major dead code issues

        return score, dead_code_count

    def _calculate_duplication_score(
        self, audit_results: Dict[str, Any]
    ) -> Tuple[float, int]:
        """Calculate duplication score (placeholder - would need duplication analysis)"""
        # For now, assume no duplication analysis available
        # In a real implementation, this would analyze code duplication
        duplication_issues = 0
        score = 1.0  # Assume good until duplication analysis is added

        return score, duplication_issues

    def _calculate_coverage_score(
        self, audit_results: Dict[str, Any]
    ) -> Tuple[float, int]:
        """Calculate coverage score (placeholder - would need coverage data)"""
        # For now, assume 80% coverage as baseline
        # In a real implementation, this would read actual coverage data
        assumed_coverage = 80  # percentage
        coverage_gap = 100 - assumed_coverage

        # Coverage score: 1.0 for 90%+ coverage, decreasing with lower coverage
        if assumed_coverage >= 90:
            score = 1.0
        elif assumed_coverage >= 80:
            score = 0.8
        elif assumed_coverage >= 70:
            score = 0.6
        elif assumed_coverage >= 60:
            score = 0.4
        else:
            score = 0.2

        return score, coverage_gap

    def _determine_severity_level(self, score: float) -> SeverityLevel:
        """Determine severity level based on score"""
        if score >= self.POLICY_THRESHOLDS["critical"]:
            return SeverityLevel.CRITICAL
        elif score >= self.POLICY_THRESHOLDS["high"]:
            return SeverityLevel.HIGH
        elif score >= self.POLICY_THRESHOLDS["medium"]:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _generate_recommendations(
        self, components: List[ScoreComponent], severity: SeverityLevel
    ) -> List[str]:
        """Generate actionable recommendations based on scores"""
        recommendations = []

        # Sort components by weighted score (worst first)
        sorted_components = sorted(components, key=lambda c: c.weighted_score)

        for component in sorted_components:
            if component.raw_score < 0.8:  # Only recommend for components with issues
                if component.name == "security":
                    recommendations.append(
                        "🔒 Review and fix security issues identified by bandit"
                    )
                    recommendations.append(
                        "🚨 Address high-severity security vulnerabilities immediately"
                    )
                elif component.name == "lint":
                    recommendations.append(
                        "🔧 Run `make lint-fix` to auto-fix formatting and import issues"
                    )
                    recommendations.append(
                        "📝 Ensure all linting tools pass before merging"
                    )
                elif component.name == "complexity":
                    recommendations.append(
                        "📊 Refactor functions with high cyclomatic complexity (>20)"
                    )
                    recommendations.append(
                        "🔄 Break down complex functions into smaller, focused functions"
                    )
                elif component.name == "dead_code":
                    recommendations.append(
                        "🧹 Remove or properly handle dead/unused code"
                    )
                    recommendations.append(
                        "🔍 Review vulture findings for false positives"
                    )
                elif component.name == "duplication":
                    recommendations.append(
                        "🔄 Extract common code into shared functions/utilities"
                    )
                    recommendations.append(
                        "📋 Consider using DRY principles to reduce duplication"
                    )
                elif component.name == "coverage":
                    recommendations.append("🧪 Increase test coverage to at least 80%")
                    recommendations.append("📈 Add unit tests for uncovered code paths")

        # Add severity-specific recommendations
        if severity == SeverityLevel.CRITICAL:
            recommendations.insert(
                0, "🚨 CRITICAL: Address all major issues before proceeding"
            )
        elif severity == SeverityLevel.HIGH:
            recommendations.insert(
                0, "⚠️ HIGH PRIORITY: Fix critical issues in next sprint"
            )
        elif severity == SeverityLevel.MEDIUM:
            recommendations.insert(0, "📋 MEDIUM: Address issues when convenient")
        else:
            recommendations.insert(0, "✅ LOW: Code quality is acceptable")

        return recommendations[:10]  # Limit to top 10 recommendations

    def _generate_policy_actions(self, severity: SeverityLevel) -> List[str]:
        """Generate policy actions based on severity level"""
        if severity == SeverityLevel.CRITICAL:
            return [
                "🚫 Block merge until issues are resolved",
                "📧 Notify maintainers immediately",
                "🔒 Require security review for critical vulnerabilities",
                "📋 Create high-priority issues for all findings",
            ]
        elif severity == SeverityLevel.HIGH:
            return [
                "💬 Add PR comments with issue details",
                "📋 Create GitHub issues for tracking",
                "👥 Request review from senior developers",
                "📅 Schedule fixes for next development cycle",
            ]
        elif severity == SeverityLevel.MEDIUM:
            return [
                "⚠️ Show warnings in CI/CD pipeline",
                "📝 Add issues to backlog",
                "👀 Optional review by team lead",
                "📊 Track in code quality metrics",
            ]
        else:  # LOW
            return [
                "ℹ️ Log informational messages",
                "📊 Include in quality reports",
                "✅ Allow merge to proceed",
                "📈 Monitor trends over time",
            ]

    def _create_breakdown(
        self, components: List[ScoreComponent], audit_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed breakdown of the scoring"""
        return {
            "component_scores": {
                comp.name: {
                    "raw_score": round(comp.raw_score, 3),
                    "weighted_score": round(comp.weighted_score, 3),
                    "weight": comp.weight,
                    "issues_count": comp.issues_count,
                    "description": comp.description,
                }
                for comp in components
            },
            "weights_used": self.weights,
            "thresholds_used": self.POLICY_THRESHOLDS,
            "audit_summary": {
                "total_files": audit_results.get("static_analysis", {})
                .get("overall_summary", {})
                .get("total_files_analyzed", 0),
                "total_functions": audit_results.get("static_analysis", {})
                .get("overall_summary", {})
                .get("complexity_issues", {})
                .get("total_functions", 0),
                "security_issues": audit_results.get("static_analysis", {})
                .get("overall_summary", {})
                .get("security_findings", {})
                .get("total_issues", 0),
                "lint_failures": audit_results.get("linting", {})
                .get("summary", {})
                .get("failed", 0),
            },
        }


class SeverityConfig:
    """Configuration for severity scoring"""

    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "weights": SeverityScorer.DEFAULT_WEIGHTS,
            "thresholds": SeverityScorer.POLICY_THRESHOLDS,
            "enabled_components": [
                "security",
                "lint",
                "complexity",
                "dead_code",
                "duplication",
                "coverage",
            ],
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except (json.JSONDecodeError, IOError):
                pass  # Use defaults if config file is invalid

        return default_config

    def create_scorer(self) -> SeverityScorer:
        """Create a scorer with the loaded configuration"""
        return SeverityScorer(self.config["weights"])


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Severity Scoring - Calculate code health scores"
    )
    parser.add_argument("--audit-results", help="Path to audit results JSON file")
    parser.add_argument("--config", help="Path to severity configuration file")
    parser.add_argument("--output", help="Output file for detailed results")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    args = parser.parse_args()

    # Load configuration
    config = SeverityConfig(args.config)
    scorer = config.create_scorer()

    # Load audit results
    if args.audit_results:
        with open(args.audit_results, "r") as f:
            audit_results = json.load(f)
    else:
        # Run audit to get results
        from .code_quality_report import CodeQualityReport

        reporter = CodeQualityReport()
        audit_results = reporter.generate_full_report()

    # Calculate health score
    health_score = scorer.calculate_health_score(audit_results)

    # Output results
    if args.format == "json":
        result = {
            "overall_score": health_score.overall_score,
            "severity_level": health_score.severity_level.value,
            "recommendations": health_score.recommendations,
            "policy_actions": health_score.policy_actions,
            "breakdown": health_score.breakdown,
        }

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
    else:
        # Text format
        print("🔍 Code Quality Health Score")
        print("=" * 40)
        print(f"Overall Score: {health_score.overall_score}/100")
        print(f"Severity Level: {health_score.severity_level.value.upper()}")
        print()

        print("📊 Component Breakdown:")
        for comp in health_score.components:
            print(".1f")
        print()

        print("💡 Recommendations:")
        for rec in health_score.recommendations:
            print(f"  • {rec}")
        print()

        print("🚨 Policy Actions:")
        for action in health_score.policy_actions:
            print(f"  • {action}")

        if args.output:
            # Save detailed results
            result = {
                "overall_score": health_score.overall_score,
                "severity_level": health_score.severity_level.value,
                "recommendations": health_score.recommendations,
                "policy_actions": health_score.policy_actions,
                "breakdown": health_score.breakdown,
            }
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n📄 Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
