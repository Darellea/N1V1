"""Static Analysis - Code complexity, security, and dead code analysis"""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple


class StaticAnalysis:
    """Runs static analysis tools and aggregates results"""

    def __init__(self, output_dir: str = "reports/audit_tmp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._common_excludes = [
            "venv",
            ".venv",
            "__pycache__",
            ".git",
            "node_modules",
            ".tox",
            "htmlcov",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
        ]

    def run_command(
        self, cmd: List[str], cwd: str = None, timeout: int = 120
    ) -> Tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout running: {' '.join(cmd)}")
            return -1, "", "Command timed out"
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
            return -1, "", f"Command not found: {cmd[0]}"
        except Exception as e:
            print(f"❌ Error running {cmd[0]}: {str(e)}")
            return -1, "", str(e)

    def run_radon_cc(self, target: str = ".") -> Dict[str, Any]:
        """Run radon cyclomatic complexity analysis"""
        print("  📊 Running cyclomatic complexity analysis...")
        output_file = self.output_dir / "radon_cc.json"

        # Use .gitignore-style exclusions for better performance
        cmd = ["radon", "cc", "-s", target, "-j", "--ignore"] + [
            f"*{exclude}*" for exclude in self._common_excludes
        ]

        returncode, stdout, stderr = self.run_command(cmd, timeout=120)

        if returncode == 0 and stdout.strip():
            try:
                with open(output_file, "w") as f:
                    f.write(stdout)
                data = json.loads(stdout)
                return {
                    "status": "success",
                    "data": data,
                    "summary": self._summarize_radon_cc(data),
                }
            except json.JSONDecodeError:
                return {"status": "error", "message": "Failed to parse radon cc JSON"}
        else:
            return {"status": "error", "message": stderr or "radon cc failed"}

    def run_radon_mi(self, target: str = ".") -> Dict[str, Any]:
        """Run radon maintainability index analysis"""
        print("  📈 Running maintainability index analysis...")
        output_file = self.output_dir / "radon_mi.txt"

        cmd = ["radon", "mi", target, "--ignore"] + [
            f"*{exclude}*" for exclude in self._common_excludes
        ]

        returncode, stdout, stderr = self.run_command(cmd, timeout=120)

        if returncode == 0 and stdout.strip():
            with open(output_file, "w") as f:
                f.write(stdout)
            return {
                "status": "success",
                "data": stdout,
                "summary": self._summarize_radon_mi(stdout),
            }
        else:
            return {"status": "error", "message": stderr or "radon mi failed"}

    def run_bandit(self, target: str = ".") -> Dict[str, Any]:
        """Run bandit security analysis with better timeout handling"""
        print("  🔒 Running security analysis...")
        output_file = self.output_dir / "bandit.json"

        # Use bandit's native exclusion with quiet mode and skip large files
        exclude_dirs = ",".join(self._common_excludes)
        cmd = [
            "bandit",
            "-r",
            target,
            "-f",
            "json",
            "-o",
            str(output_file),
            "-q",  # Quiet mode
            "-x",
            exclude_dirs,  # Exclude directories
            "--skip",
            "*/test_*,*/tests/*,*/migrations/*",  # Skip test files
        ]

        returncode, stdout, stderr = self.run_command(
            cmd, timeout=90
        )  # Reduced timeout

        # Bandit returns 1 when issues are found (normal)
        if returncode in [0, 1] and output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return {
                    "status": "success",
                    "data": data,
                    "summary": self._summarize_bandit(data),
                }
            except (json.JSONDecodeError, FileNotFoundError):
                return {"status": "error", "message": "Failed to parse bandit JSON"}
        else:
            # Return empty results on timeout/error
            return {
                "status": "partial",
                "data": {},
                "summary": {"total_issues": 0, "severity_breakdown": {}},
            }

    def run_vulture(
        self, target: str = ".", min_confidence: int = 60
    ) -> Dict[str, Any]:
        """Run vulture dead code detection with better exclusions"""
        print("  🦅 Running dead code analysis...")
        output_file = self.output_dir / "vulture.txt"

        # Build exclude patterns for vulture
        exclude_patterns = []
        for exclude in self._common_excludes + [
            "*/test_*",
            "*/tests/*",
            "*/migrations/*",
        ]:
            exclude_patterns.extend(["--exclude", f"*{exclude}*"])

        cmd = [
            "vulture",
            target,
            "--min-confidence",
            str(min_confidence),
        ] + exclude_patterns

        returncode, stdout, stderr = self.run_command(cmd, timeout=60)

        # Save output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(stdout)
            if stderr:
                f.write(f"\nSTDERR:\n{stderr}")

        # Vulture returns 1 when dead code found (normal)
        if returncode in [0, 1]:
            return {
                "status": "success",
                "data": stdout,
                "summary": self._summarize_vulture(stdout),
            }
        else:
            # Return empty results on timeout/error
            return {
                "status": "partial",
                "data": "",
                "summary": {"potential_dead_code": 0},
            }

    def run_dependency_audit(
        self, requirements_file: str = "requirements.txt"
    ) -> Dict[str, Any]:
        """Run dependency vulnerability audit using pip-audit or safety"""
        output_file = self.output_dir / "pip_audit.json"

        # Try pip-audit first
        cmd = ["pip-audit", "-r", requirements_file, "-f", "json"]
        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            try:
                # Save raw output
                with open(output_file, "w") as f:
                    f.write(stdout)

                data = json.loads(stdout)
                return {
                    "status": "success",
                    "tool": "pip-audit",
                    "data": data,
                    "summary": self._summarize_pip_audit(data),
                }
            except json.JSONDecodeError:
                pass

        # Fallback to safety
        cmd = ["safety", "check", "--json", "-r", requirements_file]
        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            try:
                with open(output_file, "w") as f:
                    f.write(stdout)

                data = json.loads(stdout)
                return {
                    "status": "success",
                    "tool": "safety",
                    "data": data,
                    "summary": self._summarize_safety(data),
                }
            except json.JSONDecodeError:
                pass

        return {"status": "error", "message": "Both pip-audit and safety failed"}

    def analyze_codebase(
        self,
        target: str = ".",
        requirements_file: str = "requirements.txt",
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """Run all static analysis tools and return structured results"""
        print("🚀 Starting static analysis...")
        start_time = time.time()

        if parallel:
            # Run compatible tools in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.run_radon_cc, target): "cyclomatic_complexity",
                    executor.submit(self.run_radon_mi, target): "maintainability_index",
                    executor.submit(self.run_bandit, target): "security_issues",
                    executor.submit(self.run_vulture, target): "dead_code",
                }

                results = {}
                for future in as_completed(futures):
                    tool_name = futures[future]
                    try:
                        results[tool_name] = future.result()
                        print(f"  ✅ {tool_name} completed")
                    except Exception as e:
                        results[tool_name] = {"status": "error", "message": str(e)}
                        print(f"  ❌ {tool_name} failed: {str(e)}")

                # Run dependency audit separately (it's I/O bound)
                results["dependency_vulnerabilities"] = self.run_dependency_audit(
                    requirements_file
                )
        else:
            # Sequential execution (fallback)
            results = {
                "cyclomatic_complexity": self.run_radon_cc(target),
                "maintainability_index": self.run_radon_mi(target),
                "security_issues": self.run_bandit(target),
                "dead_code": self.run_vulture(target),
                "dependency_vulnerabilities": self.run_dependency_audit(
                    requirements_file
                ),
            }

        # Create overall summary
        summary = {
            "total_files_analyzed": self._count_python_files(target),
            "complexity_issues": results["cyclomatic_complexity"].get("summary", {}),
            "maintainability_score": results["maintainability_index"].get(
                "summary", {}
            ),
            "security_findings": results["security_issues"].get("summary", {}),
            "dead_code_findings": results["dead_code"].get("summary", {}),
            "vulnerability_count": results["dependency_vulnerabilities"]
            .get("summary", {})
            .get("total_vulnerabilities", 0),
            "analysis_duration": round(time.time() - start_time, 2),
        }

        results["overall_summary"] = summary
        return results

    def _count_python_files(self, target: str) -> int:
        """Count Python files in target directory with proper exclusions"""
        count = 0
        target_path = Path(target)

        for root, dirs, files in os.walk(target_path):
            # Skip excluded directories more aggressively
            dirs[:] = [d for d in dirs if self._should_exclude_dir(d)]

            # Count Python files
            for file in files:
                if file.endswith(".py") and not self._should_exclude_file(
                    Path(root) / file
                ):
                    count += 1

            # Early exit if we're scanning too many files (likely including dependencies)
            if count > 1000:
                print(f"  ⚠️  Large codebase detected: {count}+ files")
                break

        return count

    def _should_exclude_dir(self, dir_name: str) -> bool:
        """Check if directory should be excluded"""
        return not (
            dir_name.startswith(".")
            or dir_name in self._common_excludes
            or any(exclude in dir_name for exclude in ["cache", "temp", "tmp"])
        )

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded"""
        return any(exclude in str(file_path) for exclude in self._common_excludes)

    def _summarize_radon_cc(self, data: Dict) -> Dict[str, Any]:
        """Summarize radon cc results"""
        summary = {"total_functions": 0, "complexity_distribution": {}}

        for file_path, functions in data.items():
            if isinstance(functions, list):
                for func_data in functions:
                    if isinstance(func_data, dict) and "complexity" in func_data:
                        complexity = func_data.get("complexity", 0)
                        summary["total_functions"] += 1

                        # Categorize complexity
                        if complexity <= 5:
                            cat = "simple"
                        elif complexity <= 10:
                            cat = "moderate"
                        elif complexity <= 20:
                            cat = "complex"
                        else:
                            cat = "very_complex"

                        summary["complexity_distribution"][cat] = (
                            summary["complexity_distribution"].get(cat, 0) + 1
                        )

        return summary

    def _summarize_radon_mi(self, output: str) -> Dict[str, Any]:
        """Summarize radon mi results"""
        lines = output.strip().split("\n")
        scores = []

        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        score = float(parts[-1])
                        scores.append(score)
                    except ValueError:
                        continue

        if scores:
            return {
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "files_analyzed": len(scores),
            }
        return {"files_analyzed": 0}

    def _summarize_bandit(self, data: Dict) -> Dict[str, Any]:
        """Summarize bandit results"""
        results = data.get("results", [])
        summary = {"total_issues": 0, "severity_breakdown": {}}

        for result in results:
            issues = result.get("issues", [])
            summary["total_issues"] += len(issues)

            for issue in issues:
                severity = issue.get("issue_severity", "unknown")
                summary["severity_breakdown"][severity] = (
                    summary["severity_breakdown"].get(severity, 0) + 1
                )

        return summary

    def _summarize_vulture(self, output: str) -> Dict[str, Any]:
        """Summarize vulture results"""
        lines = output.strip().split("\n")
        dead_items = [
            line for line in lines if line.strip() and not line.startswith(" ")
        ]

        return {"potential_dead_code": len(dead_items)}

    def _summarize_pip_audit(self, data: List) -> Dict[str, Any]:
        """Summarize pip-audit results"""
        return {"total_vulnerabilities": len(data)}

    def _summarize_safety(self, data: List) -> Dict[str, Any]:
        """Summarize safety results"""
        return {"total_vulnerabilities": len(data)}


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Static Analysis - Run code quality checks"
    )
    parser.add_argument("--target", default=".", help="Target directory")
    parser.add_argument(
        "--requirements", default="requirements.txt", help="Requirements file"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Run tools sequentially"
    )

    args = parser.parse_args()

    analyzer = StaticAnalysis()
    results = analyzer.analyze_codebase(
        args.target, args.requirements, parallel=not args.no_parallel
    )

    # Print summary
    summary = results.get("overall_summary", {})
    print(f"\n📋 Analysis Complete in {summary.get('analysis_duration', 0)}s:")
    print(f"   📁 Files analyzed: {summary.get('total_files_analyzed', 0)}")
    print(
        f"   📊 Functions analyzed: {summary.get('complexity_issues', {}).get('total_functions', 0)}"
    )
    print(
        f"   🔒 Security issues: {summary.get('security_findings', {}).get('total_issues', 0)}"
    )
    print(
        f"   🦅 Potential dead code: {summary.get('dead_code_findings', {}).get('potential_dead_code', 0)}"
    )
    print(f"   📦 Dependency vulnerabilities: {summary.get('vulnerability_count', 0)}")


if __name__ == "__main__":
    main()
