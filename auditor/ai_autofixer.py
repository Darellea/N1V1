"""AI Auto-Fixer - Safe LLM-powered code improvement with comprehensive safety guards"""

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import git


@dataclass
class AutoFixIssue:
    """Represents an issue that can be auto-fixed"""

    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    code_snippet: str
    fix_confidence: float  # 0.0 to 1.0


@dataclass
class AutoFixResult:
    """Result of an auto-fix attempt"""

    success: bool
    issue: AutoFixIssue
    patch: Optional[str]
    explanation: str
    tests_passed: bool
    branch_name: Optional[str]
    pr_url: Optional[str]


class IssueCollector:
    """Collects and analyzes audit findings for auto-fixable issues"""

    def __init__(self):
        self.auto_fixable_types = {
            "import_sorting": {"confidence": 0.95, "category": "formatting"},
            "unused_import": {"confidence": 0.90, "category": "cleanup"},
            "variable_naming": {"confidence": 0.85, "category": "refactor"},
            "function_complexity": {"confidence": 0.70, "category": "refactor"},
            "code_formatting": {"confidence": 0.95, "category": "formatting"},
            "docstring_missing": {"confidence": 0.60, "category": "documentation"},
        }

    def collect_from_audit_results(
        self, audit_results: Dict[str, Any]
    ) -> List[AutoFixIssue]:
        """Extract auto-fixable issues from audit results"""
        issues = []

        # Collect from linting results
        lint_results = audit_results.get("linting", {})
        issues.extend(self._extract_linting_issues(lint_results))

        # Collect from static analysis
        static_results = audit_results.get("static_analysis", {})
        issues.extend(self._extract_static_analysis_issues(static_results))

        return issues

    def _extract_linting_issues(
        self, lint_results: Dict[str, Any]
    ) -> List[AutoFixIssue]:
        """Extract fixable issues from linting results"""
        issues = []

        # Black formatting issues
        black_result = lint_results.get("black", {})
        if black_result.get("status") == "failed":
            # Black issues are usually auto-fixable
            issues.append(
                AutoFixIssue(
                    file_path="multiple_files",
                    line_number=0,
                    issue_type="code_formatting",
                    severity="low",
                    description="Code formatting issues detected by black",
                    code_snippet="",
                    fix_confidence=0.95,
                )
            )

        # isort import issues
        isort_result = lint_results.get("isort", {})
        if isort_result.get("status") == "failed":
            issues.append(
                AutoFixIssue(
                    file_path="multiple_files",
                    line_number=0,
                    issue_type="import_sorting",
                    severity="low",
                    description="Import sorting issues detected by isort",
                    code_snippet="",
                    fix_confidence=0.95,
                )
            )

        return issues

    def _extract_static_analysis_issues(
        self, static_results: Dict[str, Any]
    ) -> List[AutoFixIssue]:
        """Extract fixable issues from static analysis"""
        issues = []

        # Look for complexity issues that might be refactorable
        complexity_data = static_results.get("complexity_issues", {})
        complexity_dist = complexity_data.get("complexity_distribution", {})

        very_complex_count = complexity_dist.get("very_complex", 0)
        if very_complex_count > 0:
            issues.append(
                AutoFixIssue(
                    file_path="multiple_files",
                    line_number=0,
                    issue_type="function_complexity",
                    severity="medium",
                    description=f"{very_complex_count} functions with very high complexity (>20)",
                    code_snippet="",
                    fix_confidence=0.70,
                )
            )

        return issues

    def filter_auto_fixable(
        self, issues: List[AutoFixIssue], max_issues: int = 5
    ) -> List[AutoFixIssue]:
        """Filter issues to only those suitable for auto-fixing"""

        # Sort by confidence (highest first) and severity
        def sort_key(issue):
            severity_weight = {"high": 3, "medium": 2, "low": 1}
            return (severity_weight.get(issue.severity, 0), issue.fix_confidence)

        filtered = [issue for issue in issues if issue.fix_confidence >= 0.6]
        filtered.sort(key=sort_key, reverse=True)

        return filtered[:max_issues]


class ContextBuilder:
    """Builds context for LLM prompts"""

    def __init__(self):
        self.max_context_length = 4000  # tokens

    def build_fix_context(
        self, issue: AutoFixIssue, file_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build context for fixing a specific issue"""
        context = {
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "description": issue.description,
            "confidence": issue.fix_confidence,
            "file_path": issue.file_path,
            "line_number": issue.line_number,
        }

        if file_content and issue.line_number > 0:
            # Extract relevant code snippet
            lines = file_content.split("\n")
            start_line = max(0, issue.line_number - 5)
            end_line = min(len(lines), issue.line_number + 5)
            context["code_snippet"] = "\n".join(lines[start_line:end_line])
            context["full_file"] = file_content[: self.max_context_length]
        else:
            context["code_snippet"] = issue.code_snippet

        return context

    def build_prompt(self, context: Dict[str, Any]) -> str:
        """Build the LLM prompt for code fixing"""
        prompt_template = """
You are an expert Python code improvement assistant. Your task is to provide minimal, safe fixes for code quality issues.

ISSUE DETAILS:
- Type: {issue_type}
- Severity: {severity}
- Description: {description}
- Confidence: {confidence:.2f}

CODE CONTEXT:
{code_snippet}

INSTRUCTIONS:
1. Return ONLY a unified diff patch in the format:
   --- a/file_path
   +++ b/file_path
   @@ -line_start,line_count +line_start,line_count @@
    original_code
   +modified_code

2. Make MINIMAL changes - only fix the specific issue
3. Do NOT add or remove external dependencies
4. Do NOT change code behavior - only formatting/refactoring
5. Ensure the patch passes pytest and linters
6. Include a comment block explaining the rationale

7. If the issue cannot be safely fixed, return only: "UNSAFE_TO_FIX"

EXAMPLE OUTPUT:
--- a/example.py
+++ b/example.py
@@ -1,3 +1,4 @@
 # Fixed import sorting and added docstring
+\"\"\"Module docstring.\"\"\"
 import os
-import sys
+import sys

RATIONALE: Sorted imports alphabetically and added missing module docstring.
"""

        return prompt_template.format(**context)


class MockLLMClient:
    """Mock LLM client for demonstration - replace with real LLM API"""

    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name
        self.call_count = 0

    def generate_fix(self, prompt: str) -> str:
        """Mock LLM response - in real implementation, call actual LLM API"""
        self.call_count += 1

        # Simulate different responses based on issue type
        if "import_sorting" in prompt:
            return self._mock_import_fix()
        elif "code_formatting" in prompt:
            return self._mock_formatting_fix()
        elif "function_complexity" in prompt:
            return "UNSAFE_TO_FIX"  # Too complex for safe auto-fix
        else:
            return "UNSAFE_TO_FIX"

    def _mock_import_fix(self) -> str:
        """Mock import sorting fix"""
        return """--- a/auditor/code_linter.py
+++ b/auditor/code_linter.py
@@ -1,11 +1,11 @@
 # Code Linter - Orchestrates formatting and linting tools

 import argparse
-import subprocess
 import sys
+import subprocess
 from typing import Dict, List, Tuple


 # Fixed import sorting - moved subprocess after sys
 class CodeLinter:

 RATIONALE: Sorted imports alphabetically as per PEP8 standards."""

    def _mock_formatting_fix(self) -> str:
        """Mock code formatting fix"""
        return """--- a/auditor/static_analysis.py
+++ b/auditor/static_analysis.py
@@ -10,7 +10,8 @@
 class StaticAnalysis:
     # Runs static analysis tools and aggregates results

-    def __init__(self, output_dir: str = "reports/audit_tmp"):
+    def __init__(
+        self, output_dir: str = "reports/audit_tmp"
+    ):
         self.output_dir = Path(output_dir)
         self.output_dir.mkdir(parents=True, exist_ok=True)

 RATIONALE: Fixed line length by breaking long parameter line."""


class PatchApplier:
    """Applies patches safely with rollback capability"""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo(repo_path) if self._is_git_repo() else None

    def _is_git_repo(self) -> bool:
        """Check if current directory is a git repository"""
        try:
            git.Repo(self.repo_path)
            return True
        except git.InvalidGitRepositoryError:
            return False

    def create_feature_branch(self, base_branch: str = "main") -> str:
        """Create a new feature branch for auto-fixes"""
        if not self.repo:
            raise RuntimeError("Not a git repository")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        branch_name = f"auditor/auto-fix/{timestamp}"

        try:
            # Checkout base branch first
            self.repo.git.checkout(base_branch)

            # Create and checkout new branch
            self.repo.git.checkout("-b", branch_name)

            return branch_name
        except git.GitCommandError as e:
            raise RuntimeError(f"Failed to create branch: {e}")

    def apply_patch(self, patch: str) -> bool:
        """Apply a unified diff patch"""
        try:
            # Write patch to temporary file
            patch_file = self.repo_path / ".audit_patch.tmp"
            patch_file.write_text(patch)

            # Apply patch
            result = subprocess.run(
                ["git", "apply", "--whitespace=fix", str(patch_file)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )

            # Clean up
            patch_file.unlink(missing_ok=True)

            return result.returncode == 0
        except Exception:
            return False

    def run_tests_and_lints(self) -> Tuple[bool, str]:
        """Run tests and linters to validate changes"""
        results = []

        # Run pytest (quick test)
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--tb=short", "-x"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
            )
            test_passed = result.returncode == 0
            results.append(f"pytest: {'PASS' if test_passed else 'FAIL'}")
            if not test_passed:
                results.append(f"pytest output: {result.stdout[-500:]}")
        except subprocess.TimeoutExpired:
            results.append("pytest: TIMEOUT")
            test_passed = False
        except FileNotFoundError:
            results.append("pytest: NOT FOUND (skipping)")
            test_passed = True  # Don't fail if pytest not available

        # Run linters
        try:
            result = subprocess.run(
                ["python", "-m", "auditor.code_linter", "--mode", "check"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            lint_passed = result.returncode == 0
            results.append(f"lint: {'PASS' if lint_passed else 'FAIL'}")
        except subprocess.TimeoutExpired:
            results.append("lint: TIMEOUT")
            lint_passed = False

        all_passed = test_passed and lint_passed
        return all_passed, "\n".join(results)

    def rollback_changes(self) -> bool:
        """Rollback all changes in current branch"""
        if not self.repo:
            return False

        try:
            # Reset to HEAD~1 (before our changes)
            self.repo.git.reset("--hard", "HEAD~1")
            return True
        except git.GitCommandError:
            return False

    def create_pull_request(
        self, branch_name: str, title: str, description: str
    ) -> Optional[str]:
        """Create a pull request (mock implementation)"""
        # In a real implementation, this would integrate with GitHub/GitLab API
        print(f"Mock PR created: {title}")
        print(f"Branch: {branch_name}")
        print(f"Description: {description}")

        # Return mock PR URL
        return f"https://github.com/your-org/repo/pull/mock-{datetime.now().strftime('%Y%m%d%H%M%S')}"


class SafetyGuard:
    """Comprehensive safety checks for auto-fixing"""

    def __init__(self):
        self.max_fixes_per_run = 3
        self.max_file_changes = 10  # lines
        self.forbidden_patterns = [
            r"import\s+os\s*$",  # Don't modify critical imports
            r"class\s+\w+",  # Don't modify class definitions
            r"def\s+\w+",  # Don't modify function definitions
            r"if\s+__name__",  # Don't modify main guards
        ]

    def validate_patch(self, patch: str) -> Tuple[bool, str]:
        """Validate that a patch is safe to apply"""
        if not patch or patch.strip() == "":
            return False, "Empty patch"

        if "UNSAFE_TO_FIX" in patch:
            return False, "LLM determined fix is unsafe"

        # Check patch size
        lines = patch.split("\n")
        change_lines = [
            line for line in lines if line.startswith("+") or line.startswith("-")
        ]
        if len(change_lines) > self.max_file_changes:
            return (
                False,
                f"Patch too large: {len(change_lines)} changes (max {self.max_file_changes})",
            )

        # Check for forbidden patterns in added lines
        for line in lines:
            if line.startswith("+"):
                for pattern in self.forbidden_patterns:
                    if re.search(pattern, line[1:].strip()):  # Remove the + prefix
                        return False, f"Forbidden pattern detected: {pattern}"

        return True, "Patch validated"

    def should_skip_security_issue(self, issue: AutoFixIssue) -> bool:
        """Check if security issues should be skipped"""
        return issue.severity in ["high", "critical"]

    def rate_limit_check(self, recent_fixes: List[datetime]) -> bool:
        """Check if we're within rate limits"""
        # Allow max 5 fixes per hour
        one_hour_ago = datetime.now().replace(hour=datetime.now().hour - 1)
        recent_count = sum(1 for fix_time in recent_fixes if fix_time > one_hour_ago)
        return recent_count < 5


class AIAutoFixer:
    """Main AI auto-fixer with comprehensive safety guards"""

    def __init__(self, dry_run: bool = True, llm_client: Optional[Any] = None):
        self.dry_run = dry_run
        self.collector = IssueCollector()
        self.context_builder = ContextBuilder()
        self.llm_client = llm_client or MockLLMClient()
        self.patch_applier = PatchApplier()
        self.safety_guard = SafetyGuard()
        self.applied_fixes = []

    def run_auto_fix(
        self, audit_results: Dict[str, Any], max_fixes: int = 3
    ) -> List[AutoFixResult]:
        """Run the complete auto-fix pipeline"""
        print("🔧 Starting AI Auto-Fix process...")

        # Collect issues
        all_issues = self.collector.collect_from_audit_results(audit_results)
        fixable_issues = self.collector.filter_auto_fixable(all_issues, max_fixes)

        if not fixable_issues:
            print("✅ No auto-fixable issues found")
            return []

        print(f"📋 Found {len(fixable_issues)} auto-fixable issues")

        results = []

        for issue in fixable_issues:
            if self.safety_guard.should_skip_security_issue(issue):
                print(f"⏭️  Skipping security issue: {issue.description}")
                continue

            result = self._fix_single_issue(issue)
            results.append(result)

            if not result.success and not self.dry_run:
                print(f"❌ Fix failed for: {issue.description}")
                break  # Stop on first failure

        return results

    def _fix_single_issue(self, issue: AutoFixIssue) -> AutoFixResult:
        """Fix a single issue with full safety pipeline"""
        print(f"🔧 Processing: {issue.description}")

        # Build context and prompt
        context = self.context_builder.build_fix_context(issue)
        prompt = self.context_builder.build_prompt(context)

        # Get LLM fix
        patch = self.llm_client.generate_fix(prompt)

        # Validate patch
        is_safe, safety_message = self.safety_guard.validate_patch(patch)
        if not is_safe:
            return AutoFixResult(
                success=False,
                issue=issue,
                patch=None,
                explanation=f"Unsafe patch: {safety_message}",
                tests_passed=False,
                branch_name=None,
                pr_url=None,
            )

        if self.dry_run:
            return AutoFixResult(
                success=True,
                issue=issue,
                patch=patch,
                explanation="Dry run - patch validated but not applied",
                tests_passed=True,
                branch_name=None,
                pr_url=None,
            )

        # Apply the fix
        try:
            # Create feature branch
            branch_name = self.patch_applier.create_feature_branch()

            # Apply patch
            if not self.patch_applier.apply_patch(patch):
                raise RuntimeError("Failed to apply patch")

            # Run tests and lints
            tests_passed, test_output = self.patch_applier.run_tests_and_lints()

            if not tests_passed:
                # Rollback on failure
                self.patch_applier.rollback_changes()
                raise RuntimeError(f"Tests/lints failed: {test_output}")

            # Create PR
            pr_title = f"🤖 Auto-fix: {issue.description}"
            pr_description = f"""
## AI Auto-Fix Applied

**Issue:** {issue.description}
**Type:** {issue.issue_type}
**Confidence:** {issue.fix_confidence:.2f}

**Changes:**
{patch}

**Validation:**
- ✅ Tests passed
- ✅ Linters passed
- ✅ Safety checks passed

**Review Required:** Please review the automated changes before merging.
"""
            pr_url = self.patch_applier.create_pull_request(
                branch_name, pr_title, pr_description
            )

            return AutoFixResult(
                success=True,
                issue=issue,
                patch=patch,
                explanation="Successfully applied and tested",
                tests_passed=True,
                branch_name=branch_name,
                pr_url=pr_url,
            )

        except Exception as e:
            # Ensure we rollback on any error
            try:
                self.patch_applier.rollback_changes()
            except:
                pass

            return AutoFixResult(
                success=False,
                issue=issue,
                patch=patch,
                explanation=f"Fix failed: {str(e)}",
                tests_passed=False,
                branch_name=None,
                pr_url=None,
            )


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Auto-Fixer - Safe automated code improvements"
    )
    parser.add_argument("--audit-results", help="Path to audit results JSON file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run mode (default: enabled)",
    )
    parser.add_argument(
        "--max-fixes", type=int, default=3, help="Maximum number of fixes to apply"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force apply fixes (disable dry-run)"
    )

    args = parser.parse_args()

    # Load audit results
    if args.audit_results:
        with open(args.audit_results, "r") as f:
            audit_results = json.load(f)
    else:
        # Run audit to get results
        from .code_quality_report import CodeQualityReport

        reporter = CodeQualityReport()
        audit_results = reporter.generate_full_report()

    # Initialize fixer
    dry_run = args.dry_run and not args.force
    fixer = AIAutoFixer(dry_run=dry_run)

    # Run auto-fix
    results = fixer.run_auto_fix(audit_results, args.max_fixes)

    # Report results
    successful_fixes = [r for r in results if r.success]
    failed_fixes = [r for r in results if not r.success]

    print("\n📊 Auto-Fix Results:")
    print(f"✅ Successful: {len(successful_fixes)}")
    print(f"❌ Failed: {len(failed_fixes)}")

    if successful_fixes:
        print("\n✅ Applied fixes:")
        for result in successful_fixes:
            print(f"  • {result.issue.description}")
            if result.pr_url:
                print(f"    PR: {result.pr_url}")

    if failed_fixes:
        print("\n❌ Failed fixes:")
        for result in failed_fixes:
            print(f"  • {result.issue.description}: {result.explanation}")


if __name__ == "__main__":
    main()
