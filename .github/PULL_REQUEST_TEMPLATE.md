## Description

Brief description of the changes made in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Security enhancement
- [ ] Strategy addition/modification
- [ ] ML model update
- [ ] Infrastructure change

## Changes Made

### Code Changes
- List the specific files and changes made
- Include any new dependencies or configuration changes

### Test Coverage
- [ ] Unit tests added/updated for new functionality
- [ ] Integration tests added/updated for API/database changes
- [ ] Strategy tests added for new trading strategies
- [ ] ML tests added for model changes
- [ ] Manual testing performed and documented

### Documentation
- [ ] README updated (if applicable)
- [ ] Code comments added/updated with docstrings
- [ ] API documentation updated (if applicable)
- [ ] Strategy documentation added (docs/strategy_development.md)
- [ ] ML documentation updated (docs/ml_onboarding.md)

## Quality Assurance

### Testing Requirements
- [ ] All existing tests pass locally
- [ ] New tests added for new functionality with 95%+ coverage
- [ ] Test coverage maintained at 95%+ across all modules
- [ ] Linting passes (ruff/flake8 with project config)
- [ ] Type checking passes (mypy with strict settings)
- [ ] Security scanning passes (bandit, safety)

### CI/CD Validation
- [ ] CI pipeline passes for all supported Python versions (3.8, 3.9, 3.10, 3.11)
- [ ] Code coverage meets 95% requirement with no regressions
- [ ] Security vulnerability scanning passes
- [ ] Performance benchmarks show no degradation
- [ ] Docker build succeeds for all service images

### Deployment Considerations
- [ ] Changes tested in local Docker environment (docker-compose.dev.yml)
- [ ] Changes tested in staging/canary environment
- [ ] Rollback plan documented and tested
- [ ] Monitoring/alerting updated for new features
- [ ] Database migrations included and tested (if applicable)
- [ ] Environment variables documented

## Risk Assessment

### Impact Level
- [ ] Low - No impact on existing functionality (docs, tests, minor fixes)
- [ ] Medium - Minor changes to existing functionality (refactors, optimizations)
- [ ] High - Significant changes to core functionality (new strategies, API changes)
- [ ] Critical - Changes to trading logic, risk management, or financial calculations

### Trading Logic Impact
- [ ] No trading logic changes
- [ ] New strategy added (requires extensive testing)
- [ ] Existing strategy modified (requires regression testing)
- [ ] Risk management changes (requires stress testing)
- [ ] Order execution changes (requires integration testing)

### Rollback Plan
Describe the rollback strategy for this change:

1. **Immediate Rollback**: Steps to revert the change
2. **Data Cleanup**: Any data migration reversals needed
3. **Monitoring**: What to monitor during rollback
4. **Communication**: Stakeholders to notify

### Monitoring & Alerts
What metrics/alerts should be monitored post-deployment:

- [ ] System health metrics (CPU, memory, API response times)
- [ ] Trading metrics (signal generation, order execution)
- [ ] Performance metrics (PnL, win rate, drawdown)
- [ ] Error rates and exception monitoring
- [ ] Custom metrics for new features

## Checklist

### Code Quality
- [ ] Self-reviewed the code following project standards
- [ ] Code follows PEP 8 and project-specific style guidelines
- [ ] Commit messages are clear, descriptive, and follow conventional commits
- [ ] No sensitive information (API keys, credentials) exposed
- [ ] Dependencies updated in requirements files with pinned versions
- [ ] Breaking changes documented with migration guide

### Security
- [ ] Security implications reviewed and addressed
- [ ] Input validation implemented for all user inputs
- [ ] Authentication/authorization properly implemented
- [ ] No hardcoded secrets or credentials
- [ ] SQL injection and other injection attacks prevented

### Performance
- [ ] Performance impact assessed and optimized
- [ ] Memory usage reviewed for large datasets
- [ ] Database queries optimized (N+1 problems resolved)
- [ ] Caching implemented where appropriate
- [ ] Vectorized operations used for numerical computations

### Testing
- [ ] Unit tests cover all new functions and methods
- [ ] Integration tests cover API and database interactions
- [ ] Edge cases and error conditions tested
- [ ] Mock data used for external dependencies
- [ ] Deterministic tests (no random seeds without fixtures)

## Additional Notes

Any additional context, considerations, or notes for reviewers:

### Related Issues/PRs
- Closes #ISSUE_NUMBER
- Related to #ISSUE_NUMBER
- Depends on #PR_NUMBER

### Testing Instructions
Steps for reviewers to test the changes:

1. **Setup**: How to set up the testing environment
2. **Reproduction**: Steps to reproduce the issue (if bug fix)
3. **Verification**: How to verify the fix/feature works
4. **Edge Cases**: Any specific edge cases to test

### Screenshots/Demos
Include screenshots, API responses, or demo videos if applicable.

---

## CI/CD Pipeline Status

This PR will trigger the following CI/CD checks:

### 1. Build & Test Matrix (Python 3.8, 3.9, 3.10, 3.11)
- ✅ **Unit Tests**: Comprehensive test suite execution
- ✅ **Integration Tests**: API, database, and service interactions
- ✅ **Code Coverage**: Analysis with 95% minimum requirement
- ✅ **Linting**: ruff/flake8 with project configuration
- ✅ **Type Checking**: mypy with strict type enforcement
- ✅ **Security Scanning**: bandit and safety vulnerability checks
- ✅ **Performance Tests**: Benchmark comparisons and regression detection

### 2. ML Pipeline Validation (if ML changes)
- ✅ **Model Training**: Deterministic training with fixed seeds
- ✅ **Model Validation**: F1 score target validation (≥0.70)
- ✅ **Benchmarking**: Performance comparison with previous models
- ✅ **Reproducibility**: Environment snapshot and Git state validation

### 3. Docker & Infrastructure
- ✅ **Docker Build**: Multi-stage builds for all services
- ✅ **Container Tests**: Health checks and service dependencies
- ✅ **Security Scan**: Container image vulnerability scanning

### 4. Canary Deployment (main branch only)
- ✅ **Automated Deployment**: Staging environment deployment
- ✅ **Smoke Tests**: Basic functionality validation
- ✅ **Health Checks**: Service health and readiness verification
- ✅ **Automatic Rollback**: Failure-triggered rollback procedures

### Coverage Requirements
- **Minimum 95% code coverage** required across all modules
- **Core modules** (core/, api/, ml/): 100% coverage required
- **Critical paths**: Trading logic, risk management, order execution
- Coverage report generated and uploaded as CI artifact
- Coverage must include: `core`, `api`, `notifier`, `utils`, `ml`, `backtest`, `data`, `optimization`, `portfolio`

### Testing Standards
- All tests must pass without flakiness
- New features require corresponding unit and integration tests
- API changes require comprehensive integration tests
- Performance changes require benchmark tests
- ML changes require reproducibility and benchmarking tests
- Security changes require penetration testing

### Deployment Notes
- Changes deployed to canary environment first (10% traffic)
- Smoke tests validate basic functionality in canary
- Full production deployment requires manual approval
- Rollback procedures tested and documented
- Monitoring dashboards updated for new metrics

---

## Review Guidelines for Reviewers

### For Code Reviewers
1. **Check the risk assessment** - understand the impact level
2. **Review test coverage** - ensure adequate testing for changes
3. **Verify security** - check for vulnerabilities and best practices
4. **Test the changes** - follow testing instructions provided
5. **Check performance** - review for potential bottlenecks
6. **Validate documentation** - ensure docs are updated

### For QA Reviewers
1. **Test in staging** - validate in canary deployment
2. **Check monitoring** - verify metrics and alerts work
3. **Test edge cases** - validate error handling and boundaries
4. **Performance validation** - check for regressions
5. **Integration testing** - verify with external systems

### For Product Reviewers
1. **Feature validation** - ensure requirements are met
2. **User experience** - review from end-user perspective
3. **Business impact** - assess business value and risk
4. **Documentation** - verify user-facing docs are clear

---

**By submitting this PR, I confirm that:**
- ✅ All CI/CD checks will pass
- ✅ Code coverage meets the 95% requirement
- ✅ Changes have been tested in staging/canary environment
- ✅ Rollback procedures are documented and tested
- ✅ No breaking changes are introduced without migration path
- ✅ Security implications have been reviewed and addressed
- ✅ Performance impact has been assessed and optimized
- ✅ All relevant documentation has been updated
