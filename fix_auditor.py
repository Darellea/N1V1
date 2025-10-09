#!/usr/bin/env python3
"""Fix the auditor module's own code quality issues"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except:
        return False

def main():
    print("🔧 Fixing Auditor module code quality...")

    # Fix black formatting
    print("1. Running black formatter...")
    if run_command([sys.executable, "-m", "black", "auditor/"]):
        print("   ✅ Black formatting applied")
    else:
        print("   ⚠️  Black had issues (this might be normal)")

    # Fix import sorting
    print("2. Running isort...")
    if run_command([sys.executable, "-m", "isort", "auditor/"]):
        print("   ✅ Imports sorted")
    else:
        print("   ⚠️  Isort had issues")

    # Fix ruff issues (auto-fixable ones)
    print("3. Running ruff auto-fixer...")
    if run_command([sys.executable, "-m", "ruff", "check", "auditor/", "--fix"]):
        print("   ✅ Ruff auto-fixes applied")
    else:
        print("   ⚠️  Ruff had issues (some may need manual fixes)")

    print("\n🎯 Now let's check the fixed code...")

    # Verify fixes
    print("4. Verifying fixes...")
    black_ok = run_command([sys.executable, "-m", "black", "--check", "auditor/"])
    isort_ok = run_command([sys.executable, "-m", "isort", "--check-only", "auditor/"])
    ruff_ok = run_command([sys.executable, "-m", "ruff", "check", "auditor/"])

    print(f"   Black: {'✅' if black_ok else '❌'}")
    print(f"   Isort: {'✅' if isort_ok else '❌'}")
    print(f"   Ruff:  {'✅' if ruff_ok else '❌'}")

if __name__ == "__main__":
    main()
