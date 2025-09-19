# N1V1 Testing Strategy and CI/CD Integration

## Overview

N1V1 implements a comprehensive testing strategy with categorized tests, automated CI/CD pipelines, and enforced quality thresholds. This document explains the testing framework, categories, and workflows.

## Test Categories

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual functions, classes, and modules in isolation
- **Scope**: Isolated components with mocked dependencies
- **Examples**:
  - `test_config_manager.py` - Configuration loading and validation
  - `test_order_manager.py` - Order processing logic
  - `test_signal_processor.py` - Signal processing algorithms
- **When to run**: On every code change, fast feedback

### Integration Tests (`tests/integration/`)
- **Purpose**: Test interactions between multiple components
- **Scope**: End-to-end workflows with real dependencies
- **Examples**:
  - `test_order_flow_integration.py` - Complete order lifecycle
  - `test_distributed_system.py` - Multi-node coordination
  - `test_ml_serving_integration.py` - ML model serving pipeline
- **When to run**: Before merges, validate component interactions

### Stress Tests (`tests/stress/`)
- **Purpose**: Test system performance under load and failure conditions
- **Scope**: High concurrency, resource limits, failure scenarios
- **Examples**:
  - `test_load.py` - 1000+ concurrent trades simulation
  - `test_cluster_scaling.py` - Multi-node scaling scenarios
  - `chaos_tests.py` - Failure injection and recovery
- **When to run**: Nightly, before releases

## Test Runner

The comprehensive test runner provides flexible execution options:

```bash
# Run all tests
python tests/run_comprehensive_tests.py

# Run only unit tests
python tests/run_comprehensive_tests.py --unit

# Run only integration tests
python tests/run_comprehensive_tests.py --integration

# Run only stress tests
python tests/run_comprehensive_tests.py --stress

# Run smoke tests for canary deployments
python tests/run_comprehensive_tests.py --smoke
```

### Smoke Tests
Smoke tests validate minimal functionality for canary deployments:
- API health checks
- Basic order flow (create → execute → track)
- Configuration loading
- Database connectivity
- **Target runtime**: < 2 minutes

## CI/CD Pipeline

### Pull Request Pipeline
1. **Linting**: Code style and type checking
2. **Unit + Integration Tests**: Fast feedback on code changes
3. **Coverage Enforcement**: Minimum 95% coverage required
4. **Security Scans**: Automated vulnerability detection

### Nightly Pipeline
1. **Full Test Suite**: All unit, integration, and stress tests
2. **Performance Benchmarks**: Latency and throughput validation
3. **Chaos Engineering**: Failure scenario testing
4. **Report Generation**: Detailed metrics and artifacts

### Canary Deployment Pipeline
1. **Smoke Tests**: Quick validation before deployment
2. **Gradual Rollout**: 10% → 25% → 50% → 100% traffic
3. **Monitoring**: Real-time metrics and rollback triggers

## Coverage Requirements

- **Minimum Coverage**: 95% across all modules
- **Critical Modules**: 98% minimum (core/, utils/)
- **Coverage Reports**: Generated for every CI run
- **Enforcement**: PRs blocked if coverage drops below threshold

### Coverage Configuration

Coverage is configured in `pytest.ini` and `.coveragerc`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = --cov=core --cov=utils --cov=api --cov=ml --cov=notifier --cov-fail-under=95

[coverage:run]
source = core, utils, api, ml, notifier
omit = */tests/*, */venv/*

[coverage:report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Running Tests Locally

### Prerequisites
```bash
pip install -r requirements-dev.txt
pip install pytest pytest-cov pytest-asyncio
```

### Development Workflow
```bash
# Run tests for specific module
pytest tests/unit/test_config_manager.py -v

# Run with coverage
pytest tests/unit/ --cov=core.config_manager --cov-report=html

# Run integration tests
pytest tests/integration/ -v

# Run stress tests (caution: resource intensive)
pytest tests/stress/test_load.py::test_load_stress_test
```

### Debug Mode
```bash
# Run with debugging
pytest tests/unit/test_order_manager.py -v -s --pdb

# Run specific test method
pytest tests/integration/test_order_flow_integration.py::TestOrderFlowIntegration::test_complete_order_flow_paper_trading -v
```

## Test Organization

### File Naming Convention
- `test_*.py` - Test files
- `Test*` - Test classes
- `test_*` - Test methods
- `mock_*` - Mock objects and fixtures

### Directory Structure
```
tests/
├── unit/                    # Unit tests
│   ├── test_config_manager.py
│   ├── test_order_manager.py
│   └── test_coverage_enforcement.py
├── integration/             # Integration tests
│   ├── test_order_flow_integration.py
│   └── test_distributed_system.py
├── stress/                  # Stress and load tests
│   ├── test_load.py
│   └── test_cluster_scaling.py
├── conftest.py             # Shared fixtures
└── run_comprehensive_tests.py  # Test runner
```

## Quality Gates

### Code Quality
- **Linting**: Ruff for style, Flake8 for complexity
- **Type Checking**: MyPy with strict mode
- **Security**: Bandit for vulnerability scanning
- **Dependencies**: Safety for known vulnerabilities

### Performance
- **Latency**: P95 < 150ms for order execution
- **Throughput**: > 100 orders/second sustained
- **Memory**: < 500MB per process
- **CPU**: < 80% utilization under load

### Reliability
- **Uptime**: 99.9% service availability
- **Error Rate**: < 0.5% failed orders
- **Recovery**: < 30 seconds from failures
- **Data Integrity**: 100% consistency guarantees

## Monitoring and Reporting

### Test Metrics
- Coverage percentage by module
- Test execution time trends
- Failure rates and patterns
- Performance benchmarks

### CI Artifacts
- Coverage reports (HTML/XML)
- Test result summaries
- Performance metrics
- Security scan results

### Notifications
- Slack/Discord alerts for failures
- Weekly summary reports
- Performance regression alerts

## Contributing

### Adding Tests
1. Identify appropriate category (unit/integration/stress)
2. Follow naming conventions
3. Add comprehensive docstrings
4. Include edge cases and error scenarios
5. Update this documentation if needed

### Test Best Practices
- Use descriptive test names
- Keep tests fast and isolated
- Mock external dependencies
- Test both success and failure paths
- Include performance assertions where relevant

### CI/CD Integration
- All tests must pass before merge
- Coverage requirements are enforced
- New tests are automatically discovered
- Failures block deployment pipelines

## Troubleshooting

### Common Issues
- **Import Errors**: Check PYTHONPATH and virtual environment
- **Mock Failures**: Verify mock setup and cleanup
- **Coverage Gaps**: Run with `--cov-report=html` for detailed reports
- **Timeout Issues**: Increase timeout or optimize test setup

### Debug Tools
- `pytest --pdb` - Interactive debugging
- `pytest --cov-report=html` - Visual coverage reports
- `pytest --durations=10` - Slowest tests identification
- `pytest --lf` - Run only failed tests

## Future Enhancements

- Property-based testing with Hypothesis
- Mutation testing for coverage validation
- AI-assisted test generation
- Distributed test execution
- Real-time performance monitoring integration
