"""Code Linter - Orchestrates formatting and linting tools"""

import argparse
import subprocess
import sys
from typing import Dict, List, Tuple


class CodeLinter:
    """Orchestrates code formatting and linting tools"""

    def __init__(self):
        self.tools = {
            "black": {
                "check": ["black", "--check", "--diff", "."],
                "fix": ["black", "."],
            },
            "isort": {
                "check": ["isort", "--check-only", "--diff", "."],
                "fix": ["isort", "."],
            },
            "ruff": {
                "check": ["ruff", "check", "."],
                "fix": ["ruff", "check", ".", "--fix"],
            },
            "flake8": {
                "check": ["flake8", "."],
                "fix": None,  # flake8 doesn't have auto-fix
            },
        }

    def run_command(self, cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",  # Force UTF-8 encoding
                errors="replace",  # Replace problematic characters
                timeout=300,  # 5 minute timeout for analysis
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd[0]}"

    def run_tool(self, tool_name: str, mode: str, target: str) -> Dict:
        """Run a single linting tool"""
        if tool_name not in self.tools:
            return {"tool": tool_name, "status": "error", "message": "Unknown tool"}

        tool_config = self.tools[tool_name]
        if mode not in tool_config:
            return {
                "tool": tool_name,
                "status": "skipped",
                "message": f"No {mode} mode for {tool_name}",
            }

        cmd = tool_config[mode]
        if cmd is None:
            return {
                "tool": tool_name,
                "status": "skipped",
                "message": f"{tool_name} does not support {mode}",
            }

        # Replace '.' with target if specified
        if target != ".":
            cmd = [arg.replace(".", target) if arg == "." else arg for arg in cmd]

        returncode, stdout, stderr = self.run_command(
            cmd, cwd=target if target != "." else None
        )

        result = {
            "tool": tool_name,
            "mode": mode,
            "command": cmd,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

        if returncode == 0:
            result["status"] = "passed"
        elif returncode == -1:
            result["status"] = "error"
        else:
            result["status"] = "failed"

        return result

    def run_linters(
        self, target: str = ".", mode: str = "check", tools: List[str] = None
    ) -> Dict:
        """Run all linters in specified mode"""
        if tools is None:
            tools = ["black", "isort", "ruff"]  # Default tools, flake8 optional

        results = {}
        summary = {"total": len(tools), "passed": 0, "failed": 0, "errors": 0}

        print(f"Running {mode} on {target} with tools: {', '.join(tools)}")

        for tool in tools:
            print(f"Running {tool}...")
            result = self.run_tool(tool, mode, target)
            results[tool] = result

            if result["status"] == "passed":
                summary["passed"] += 1
                print(f"✓ {tool} passed")
            elif result["status"] == "failed":
                summary["failed"] += 1
                print(f"✗ {tool} failed")
                if result.get("stdout"):
                    print(result["stdout"][:500])  # First 500 chars
                if result.get("stderr"):
                    print(result["stderr"][:500])
            else:
                summary["errors"] += 1
                print(f"⚠ {tool} error: {result.get('message', 'Unknown error')}")

        results["summary"] = summary
        return results

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(
            description="Code Linter - Run formatting and linting tools"
        )
        parser.add_argument("--target", default=".", help="Target directory")
        parser.add_argument(
            "--mode",
            choices=["check", "fix"],
            default="check",
            help="Mode: check or fix",
        )
        parser.add_argument(
            "--tools",
            nargs="+",
            choices=["black", "isort", "ruff", "flake8"],
            default=["black", "isort", "ruff"],
            help="Tools to run",
        )

        args = parser.parse_args()

        linter = CodeLinter()
        results = linter.run_linters(args.target, args.mode, args.tools)

        # Exit with non-zero if any failures
        if results["summary"]["failed"] > 0 or results["summary"]["errors"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    CodeLinter.main()
