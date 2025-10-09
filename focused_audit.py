#!/usr/bin/env python3
"""Focused audit script to test just the auditor module"""

import os
import sys
from pathlib import Path

def main():
    print("🎯 Running focused audit on auditor module only...")

    # Add current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))

    try:
        from auditor.code_quality_report import CodeQualityReport
        from auditor.severity import SeverityScorer

        reporter = CodeQualityReport()
        results = reporter.generate_full_report('auditor')  # Only audit auditor module

        scorer = SeverityScorer()
        health_score = scorer.calculate_health_score(results)

        print('\n🔍 AUDITOR MODULE HEALTH SCORE')
        print('=' * 50)
        print(f'Overall Score: {health_score.overall_score}/100')
        print(f'Severity Level: {health_score.severity_level.value.upper()}')
        print()

        print('📊 COMPONENT BREAKDOWN:')
        for comp in health_score.components:
            print(f'  {comp.name:12} {comp.weighted_score:5.1f} ({comp.raw_score:.2f} * {comp.weight:.2f}) - {comp.description}')

        print()
        print('💡 RECOMMENDATIONS:')
        for rec in health_score.recommendations[:5]:
            print(f'   • {rec}')

        print()
        print('🚨 POLICY ACTIONS:')
        for action in health_score.policy_actions:
            print(f'   • {action}')

    except Exception as e:
        print(f"❌ Audit failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
