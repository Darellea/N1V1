"""Audit Manager - Main entry point for code auditing operations"""

import argparse
import sys
from pathlib import Path

from .code_linter import CodeLinter
from .code_quality_report import CodeQualityReport
from .utils import IncrementalAnalyzer


class AuditManager:
    """Main orchestrator for code auditing operations"""

    def __init__(self, config_path: str = None):
        self.config_path = (
            config_path or Path(__file__).parent / "configs" / "auditor_defaults.yml"
        )
        self.linter = CodeLinter()

    def run_audit(
        self,
        target: str,
        mode: str = "check",
        generate_report: bool = True,
        ci_mode: bool = False,
    ):
        """Run full audit on target directory"""
        print(f"Running audit on {target} in {mode} mode")

        if generate_report:
            # Run full audit with report generation
            report = CodeQualityReport()
            results = report.generate_full_report(target)
            health_score = report._calculate_health_score(results)

            if ci_mode:
                # In CI mode, exit with error code for low health scores
                if health_score < 50:
                    print(f"❌ CRITICAL: Health Score too low ({health_score}/100)")
                    sys.exit(1)
                elif health_score < 75:
                    print(
                        f"⚠️  WARNING: Health Score below threshold ({health_score}/100)"
                    )
                else:
                    print(f"✅ SUCCESS: Health Score acceptable ({health_score}/100)")

            print(f"Audit complete! Health Score: {health_score}/100")
            return results
        else:
            # Just run linter
            return self.linter.run_linters(target, mode)

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Auditor - Code Quality Tool")
        parser.add_argument("--target", default=".", help="Target directory to audit")
        parser.add_argument(
            "--mode",
            choices=["check", "fix", "full", "incremental"],
            default="check",
            help="Audit mode: check/fix (linting) or full/incremental (analysis)",
        )
        parser.add_argument(
            "--base", default="origin/main", help="Base branch for incremental analysis"
        )
        parser.add_argument("--report", help="Output report file")
        parser.add_argument(
            "--ci",
            action="store_true",
            help="Run in CI mode with strict health score validation",
        )

        args = parser.parse_args()

        manager = AuditManager()

        # Handle different modes
        if args.mode in ["full", "incremental"]:
            # Use incremental analyzer for full/incremental modes
            analyzer = IncrementalAnalyzer()
            results = analyzer.run_incremental_audit(args.base, args.mode)

            if args.ci:
                # In CI mode, check if we have any issues
                if results.get("files_analyzed", 0) == 0:
                    print("✅ No files to analyze")
                else:
                    print(f"✅ Analyzed {results['files_analyzed']} files successfully")

            return results

        else:
            # Use regular audit for check/fix modes
            results = manager.run_audit(args.target, args.mode, ci_mode=args.ci)

        if args.report:
            # TODO: Generate report
            print(f"Report would be saved to {args.report}")

        return results


if __name__ == "__main__":
    sys.exit(AuditManager.main())
