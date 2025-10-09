"""Dependency Checker - Analyze package dependencies, licenses, and vulnerabilities"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DependencyChecker:
    """Check dependencies for outdated packages, licenses, and security issues"""

    def __init__(self, output_dir: str = "reports/audit_tmp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Common allowed licenses (can be configured)
        self.allowed_licenses = {
            "MIT",
            "BSD",
            "Apache-2.0",
            "ISC",
            "GPL-3.0",
            "LGPL-3.0",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "CC0-1.0",
            "Unlicense",
            "Zlib",
        }

    def run_command(self, cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd[0]}"

    def check_outdated_packages(self) -> Dict[str, Any]:
        """Check for outdated packages using pip list --outdated"""
        output_file = self.output_dir / "pip_outdated.json"
        cmd = ["pip", "list", "--outdated", "--format", "json"]

        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            try:
                data = json.loads(stdout)
                # Save raw output
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

                return {
                    "status": "success",
                    "data": data,
                    "summary": self._summarize_outdated(data),
                }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Failed to parse pip outdated JSON",
                }
        else:
            return {
                "status": "error",
                "message": stderr or "pip list --outdated failed",
            }

    def check_dependency_tree(self) -> Dict[str, Any]:
        """Check dependency tree using pipdeptree"""
        output_file = self.output_dir / "pipdeptree.json"
        cmd = ["pipdeptree", "--json"]

        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            try:
                data = json.loads(stdout)
                # Save raw output
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

                return {
                    "status": "success",
                    "data": data,
                    "summary": self._summarize_dependency_tree(data),
                }
            except json.JSONDecodeError:
                return {"status": "error", "message": "Failed to parse pipdeptree JSON"}
        else:
            return {"status": "error", "message": stderr or "pipdeptree failed"}

    def check_licenses(self) -> Dict[str, Any]:
        """Check package licenses using pip-licenses or similar"""
        output_file = self.output_dir / "licenses.json"

        # Try pip-licenses first
        cmd = ["pip-licenses", "--format", "json"]
        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            try:
                data = json.loads(stdout)
                # Save raw output
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2)

                return {
                    "status": "success",
                    "tool": "pip-licenses",
                    "data": data,
                    "summary": self._summarize_licenses(data),
                }
            except json.JSONDecodeError:
                pass

        # Fallback: try to get license info from pip show
        return self._check_licenses_fallback()

    def _check_licenses_fallback(self) -> Dict[str, Any]:
        """Fallback license checking using pip show"""
        cmd = ["pip", "list", "--format", "freeze"]
        returncode, stdout, stderr = self.run_command(cmd)

        if returncode != 0:
            return {"status": "error", "message": "Failed to get package list"}

        packages = []
        for line in stdout.strip().split("\n"):
            if "==" in line:
                package_name = line.split("==")[0]
                packages.append(package_name)

        license_data = []
        problematic_licenses = []

        for package in packages[:50]:  # Limit to first 50 packages for performance
            cmd = ["pip", "show", package]
            returncode, stdout, stderr = self.run_command(cmd)

            if returncode == 0:
                license_info = self._parse_pip_show_license(stdout)
                license_data.append({"name": package, "license": license_info})

                if license_info and license_info not in self.allowed_licenses:
                    problematic_licenses.append(
                        {"package": package, "license": license_info}
                    )

        result = {
            "status": "success",
            "tool": "pip-show-fallback",
            "data": license_data,
            "summary": {
                "total_packages_checked": len(license_data),
                "problematic_licenses": len(problematic_licenses),
                "problematic_packages": problematic_licenses,
            },
        }

        # Save output
        output_file = self.output_dir / "licenses.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def check_dependencies(
        self, requirements_file: str = "requirements.txt"
    ) -> Dict[str, Any]:
        """Run all dependency checks and return structured results"""
        print("Checking dependencies...")

        results = {
            "outdated_packages": self.check_outdated_packages(),
            "dependency_tree": self.check_dependency_tree(),
            "license_check": self.check_licenses(),
        }

        # Create overall summary
        summary = {
            "outdated_count": results["outdated_packages"]
            .get("summary", {})
            .get("total_outdated", 0),
            "total_dependencies": results["dependency_tree"]
            .get("summary", {})
            .get("total_packages", 0),
            "license_issues": results["license_check"]
            .get("summary", {})
            .get("problematic_licenses", 0),
        }

        results["overall_summary"] = summary
        return results

    def _summarize_outdated(self, data: List[Dict]) -> Dict[str, Any]:
        """Summarize outdated packages"""
        return {"total_outdated": len(data), "packages": data}

    def _summarize_dependency_tree(self, data: List[Dict]) -> Dict[str, Any]:
        """Summarize dependency tree"""
        total_packages = len(data)
        top_level_deps = sum(1 for pkg in data if not pkg.get("dependencies", []))

        return {
            "total_packages": total_packages,
            "top_level_dependencies": top_level_deps,
        }

    def _summarize_licenses(self, data: List[Dict]) -> Dict[str, Any]:
        """Summarize license information"""
        problematic_licenses = []

        for package in data:
            license_name = package.get("License", "")
            if license_name and license_name not in self.allowed_licenses:
                problematic_licenses.append(
                    {"package": package.get("Name", ""), "license": license_name}
                )

        return {
            "total_packages_checked": len(data),
            "problematic_licenses": len(problematic_licenses),
            "problematic_packages": problematic_licenses,
        }

    def _parse_pip_show_license(self, output: str) -> Optional[str]:
        """Parse license from pip show output"""
        for line in output.split("\n"):
            if line.startswith("License:"):
                return line.split(":", 1)[1].strip()
        return None


# CLI interface
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dependency Checker - Analyze package dependencies"
    )
    parser.add_argument(
        "--requirements", default="requirements.txt", help="Requirements file"
    )

    args = parser.parse_args()

    checker = DependencyChecker()
    results = checker.check_dependencies(args.requirements)

    # Print summary
    summary = results.get("overall_summary", {})
    print("Dependency check complete:")
    print(f"- Outdated packages: {summary.get('outdated_count', 0)}")
    print(f"- Total dependencies: {summary.get('total_dependencies', 0)}")
    print(f"- License issues: {summary.get('license_issues', 0)}")


if __name__ == "__main__":
    main()
