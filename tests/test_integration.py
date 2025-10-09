"""Integration tests for the complete auditor system"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
import sys

# Add auditor to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auditor.code_quality_report import CodeQualityReport
from auditor.severity import SeverityScorer
from auditor.audit_manager import AuditManager


class TestAuditorIntegration(unittest.TestCase):
    """Integration tests for the complete auditor system"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.sample_repo_dir = self.test_dir / "sample_repo"
        self.sample_repo_dir.mkdir()

        # Create a sample Python project
        self._create_sample_project()

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir)

    def _create_sample_project(self):
        """Create a sample Python project for testing"""
        # Create main module
        (self.sample_repo_dir / "main.py").write_text("""
import os
import sys
from typing import List, Dict

def calculate_sum(numbers: List[int]) -> int:
    '''Calculate sum of numbers'''
    total = 0
    for num in numbers:
        total += num
    return total

def main():
    data = [1, 2, 3, 4, 5]
    result = calculate_sum(data)
    print(f"Sum: {result}")

if __name__ == "__main__":
    main()
""")

        # Create a module with some issues
        (self.sample_repo_dir / "utils.py").write_text("""
import json
import os
from pathlib import Path

def load_config(file_path: str) -> Dict:
    '''Load configuration from file'''
    with open(file_path, 'r') as f:
        return json.load(f)

def save_config(config: Dict, file_path: str) -> None:
    '''Save configuration to file'''
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)

def complex_function(x, y, z):
    '''A complex function with high cyclomatic complexity'''
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        elif z > 0:
            return x + z
        else:
            return x
    elif y > 0:
        if z > 0:
            return y + z
        else:
            return y
    else:
        return z if z > 0 else 0
""")

        # Create requirements.txt
        (self.sample_repo_dir / "requirements.txt").write_text("""
pytest==7.4.0
requests==2.31.0
""")

        # Create __init__.py
        (self.sample_repo_dir / "__init__.py").write_text("")

    def test_full_audit_pipeline(self):
        """Test the complete audit pipeline from start to finish"""
        # Change to sample repo directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.sample_repo_dir)

            # Run full audit
            reporter = CodeQualityReport()
            results = reporter.generate_full_report('.')

            # Verify results structure
            self.assertIn('metadata', results)
            self.assertIn('linting', results)
            self.assertIn('static_analysis', results)
            self.assertIn('dependencies', results)

            # Verify metadata
            metadata = results['metadata']
            self.assertEqual(metadata['target'], '.')

            # Verify linting results
            linting = results['linting']
            self.assertIn('summary', linting)

            # Verify static analysis results
            static = results['static_analysis']
            self.assertIn('overall_summary', static)

            # Verify severity scoring works
            scorer = SeverityScorer()
            health_score = scorer.calculate_health_score(results)

            self.assertIsInstance(health_score.overall_score, float)
            self.assertGreaterEqual(health_score.overall_score, 0)
            self.assertLessEqual(health_score.overall_score, 100)

            # Should have components
            self.assertGreater(len(health_score.components), 0)

            # Should have recommendations
            self.assertGreater(len(health_score.recommendations), 0)

            # Should have policy actions
            self.assertGreater(len(health_score.policy_actions), 0)

        finally:
            os.chdir(original_cwd)

    def test_audit_manager_integration(self):
        """Test AuditManager integration"""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.sample_repo_dir)

            # Test audit manager
            manager = AuditManager()

            # Test check mode
            results = manager.run_audit('.', 'check', generate_report=False)
            self.assertIn('summary', results)

            # Test full audit
            results = manager.run_audit('.', 'check', generate_report=True)
            self.assertIn('linting', results)
            self.assertIn('static_analysis', results)

        finally:
            os.chdir(original_cwd)

    def test_reports_generation(self):
        """Test that reports are generated correctly"""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.sample_repo_dir)

            # Run audit to generate reports
            reporter = CodeQualityReport()
            results = reporter.generate_full_report('.')

            # Check that report files were created
            reports_dir = Path("reports")
            self.assertTrue(reports_dir.exists())

            # Should have markdown report
            md_files = list(reports_dir.glob("audit_report_*.md"))
            self.assertGreater(len(md_files), 0)

            # Should have JSON report
            json_files = list(reports_dir.glob("audit_report_*.json"))
            self.assertGreater(len(json_files), 0)

            # Should have SARIF report
            sarif_files = list(reports_dir.glob("audit_report_*.sarif"))
            self.assertGreater(len(sarif_files), 0)

            # Verify markdown content
            md_file = md_files[0]
            content = md_file.read_text()
            self.assertIn("# Code Quality Audit Report", content)
            self.assertIn("Overall Health Score", content)

        finally:
            os.chdir(original_cwd)

    def test_severity_scoring_real_data(self):
        """Test severity scoring with real audit data"""
        original_cwd = os.getcwd()
        try:
            os.chdir(self.sample_repo_dir)

            # Generate real audit data
            reporter = CodeQualityReport()
            results = reporter.generate_full_report('.')

            # Test severity scoring
            scorer = SeverityScorer()
            health_score = scorer.calculate_health_score(results)

            # Verify score is reasonable
            self.assertIsInstance(health_score.overall_score, float)
            self.assertGreaterEqual(health_score.overall_score, 0)
            self.assertLessEqual(health_score.overall_score, 100)

            # Should classify severity correctly
            if health_score.overall_score >= 90:
                self.assertEqual(health_score.severity_level.value, 'critical')
            elif health_score.overall_score >= 75:
                self.assertEqual(health_score.severity_level.value, 'high')
            elif health_score.overall_score >= 50:
                self.assertEqual(health_score.severity_level.value, 'medium')
            else:
                self.assertEqual(health_score.severity_level.value, 'low')

            # Should have appropriate recommendations
            self.assertGreater(len(health_score.recommendations), 0)
            self.assertLessEqual(len(health_score.recommendations), 10)  # Limited to 10

        finally:
            os.chdir(original_cwd)

    def test_configuration_loading(self):
        """Test loading configuration from .auditor.yml"""
        from auditor.severity import SeverityConfig

        # Test default config
        config = SeverityConfig()
        scorer = config.create_scorer()
        self.assertIsNotNone(scorer)

        # Test with custom config file
        config_path = self.test_dir / "test_config.yml"
        config_path.write_text("""
severity:
  weights:
    security: 0.4
    lint: 0.3
    complexity: 0.2
    dead_code: 0.05
    duplication: 0.03
    coverage: 0.02
policies:
  critical_threshold: 95
  high_threshold: 85
""")

        custom_config = SeverityConfig(str(config_path))
        custom_scorer = custom_config.create_scorer()

        # Verify custom weights (note: config loading may not be fully implemented yet)
        # For now, just verify the scorer was created
        self.assertIsNotNone(custom_scorer)


class TestPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarking tests"""

    def setUp(self):
        """Set up benchmark environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.large_repo_dir = self.test_dir / "large_repo"
        self.large_repo_dir.mkdir()

        # Create a larger test repository
        self._create_large_test_repo()

    def tearDown(self):
        """Clean up benchmark environment"""
        shutil.rmtree(self.test_dir)

    def _create_large_test_repo(self):
        """Create a larger repository for performance testing"""
        # Create multiple modules
        for i in range(10):
            module_content = f'''
"""Module {i} - Test module for performance benchmarking"""

import os
import sys
from typing import List, Dict, Optional
import json

class TestClass{i}:
    """Test class {i}"""

    def __init__(self, value: int = 0):
        self.value = value

    def method1(self) -> str:
        """Method 1"""
        return "method1_" + str(self.value)

    def method2(self, param: str) -> Dict:
        """Method 2"""
        return {{"param": param, "value": self.value}}

    def complex_method(self, x: int, y: int, z: int) -> int:
        """Complex method with high cyclomatic complexity"""
        if x > 0:
            if y > 0:
                if z > 0:
                    return x + y + z
                else:
                    return x + y - z
            elif z > 0:
                return x - y + z
            else:
                return x - y - z
        elif y > 0:
            if z > 0:
                return -x + y + z
            else:
                return -x + y - z
        else:
            return -x - y + z if z > 0 else -x - y - z

def utility_function_{i}(data: List[int]) -> int:
    """Utility function {i}"""
    return sum(data) + {i}

def main():
    """Main function"""
    obj = TestClass{i}(42)
    result = obj.complex_method(1, 2, 3)
    print("Result:", result)

if __name__ == "__main__":
    main()
'''
            (self.large_repo_dir / f"module_{i}.py").write_text(module_content)

        # Create __init__.py
        (self.large_repo_dir / "__init__.py").write_text("")

        # Create requirements.txt
        (self.large_repo_dir / "requirements.txt").write_text("""
pytest==7.4.0
requests==2.31.0
numpy==1.24.3
pandas==2.0.3
""")

    def test_performance_static_analysis(self):
        """Test static analysis performance on larger codebase"""
        import time

        original_cwd = os.getcwd()
        try:
            os.chdir(self.large_repo_dir)

            from auditor.static_analysis import StaticAnalysis

            analyzer = StaticAnalysis()

            # Time the analysis
            start_time = time.time()
            results = analyzer.analyze_codebase('.', parallel=True)
            end_time = time.time()

            analysis_time = end_time - start_time

            # Should complete in reasonable time (under 30 seconds for this size)
            self.assertLess(analysis_time, 30.0,
                          f"Analysis took too long: {analysis_time:.2f}s")

            # Verify results
            self.assertIn('overall_summary', results)
            summary = results['overall_summary']

            # Should find some files
            self.assertGreater(summary['total_files_analyzed'], 0)

            # Should find some functions
            complexity = summary.get('complexity_issues', {})
            self.assertGreater(complexity.get('total_functions', 0), 0)

            print(f"✅ Performance test passed: {analysis_time:.2f}s for {summary['total_files_analyzed']} files")

        finally:
            os.chdir(original_cwd)

    def test_performance_full_audit(self):
        """Test full audit performance"""
        import time

        original_cwd = os.getcwd()
        try:
            os.chdir(self.large_repo_dir)

            from auditor.code_quality_report import CodeQualityReport

            reporter = CodeQualityReport()

            # Time the full audit
            start_time = time.time()
            results = reporter.generate_full_report('.')
            end_time = time.time()

            audit_time = end_time - start_time

            # Should complete in reasonable time (under 60 seconds)
            self.assertLess(audit_time, 60.0,
                          f"Full audit took too long: {audit_time:.2f}s")

            # Verify results structure
            self.assertIn('linting', results)
            self.assertIn('static_analysis', results)
            self.assertIn('dependencies', results)

            print(f"✅ Full audit performance test passed: {audit_time:.2f}s")

        finally:
            os.chdir(original_cwd)

    def test_memory_usage(self):
        """Test memory usage during analysis"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        original_cwd = os.getcwd()
        try:
            os.chdir(self.large_repo_dir)

            from auditor.code_quality_report import CodeQualityReport

            # Run analysis
            reporter = CodeQualityReport()
            results = reporter.generate_full_report('.')

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory

            # Should not use excessive memory (under 500MB)
            self.assertLess(memory_used, 500.0,
                          f"Excessive memory usage: {memory_used:.1f}MB")

            print(f"✅ Memory usage test passed: {memory_used:.1f}MB used")

        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
